import Foundation
import os

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
    /// Download size shown next to catalog entries, e.g. "0.99 GB".
    var sizeLabel: String? = nil
    /// Marks the heavyweight option in the list.
    var isLarge = false
    /// Names older builds saved this file under (the old downloader cut names at the first ".").
    var legacyFilenames: [String] = []
}

/// Breadcrumb log that survives a crash: each step is written to disk before it starts, so after
/// an unexpected exit the next launch can say what the app was doing and how much memory was free.
enum Diagnostics {
    private static let maxLines = 300
    private static let url: URL = {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("diagnostics.log")
    }()

    static func mark(_ event: String) {
        let freeMB = os_proc_available_memory() / 1_048_576
        let line = "\(timestamp()) [\(freeMB) MB free] \(event)\n"
        guard let data = line.data(using: .utf8) else { return }
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(data)
            handle.synchronizeFile()
            handle.closeFile()
        } else {
            try? data.write(to: url)
        }
    }

    /// Recent log lines, newest last.
    static func recent(_ count: Int = 40) -> String {
        guard let text = try? String(contentsOf: url) else { return "" }
        return text.split(separator: "\n").suffix(count).joined(separator: "\n")
    }

    /// Called once at launch: reports a step that began but never finished in the previous run,
    /// then trims the log.
    static func startSession() -> String? {
        let previous = (try? String(contentsOf: url)) ?? ""
        let lines = previous.split(separator: "\n").map(String.init)
        var unfinished: String?
        for line in lines.reversed() {
            if line.contains("END ") || line.contains("APP START") { break }
            if line.contains("BEGIN ") { unfinished = line; break }
        }
        let kept = lines.suffix(maxLines).joined(separator: "\n")
        try? (kept.isEmpty ? "" : kept + "\n").write(to: url, atomically: true, encoding: .utf8)
        mark("APP START")
        return unfinished
    }

    private static func timestamp() -> String {
        let f = DateFormatter()
        f.dateFormat = "MM-dd HH:mm:ss"
        return f.string(from: Date())
    }
}

struct ChatMessage: Identifiable, Equatable {
    enum Role: String { case user, assistant }

    let id = UUID()
    let role: Role
    var text: String
    /// App-generated notice shown in the chat but never sent to the model.
    var isNotice = false
}

@MainActor
class LlamaState: ObservableObject {
    /// Activity log (model loading, benchmark, server); shown in Settings, not in the chat.
    @Published var messageLog = ""

    // MARK: Chat state
    @Published var messages: [ChatMessage] = []
    @Published var isGenerating = false
    @Published var loadedModelName: String?
    @Published var isLoadingModel = false
    /// Set at launch when the previous run ended in the middle of a step (likely a crash).
    @Published var previousCrashNote: String?
    /// First message that was still inside the assistant's memory for the latest reply.
    @Published var memoryStartID: UUID?
    /// Bumped whenever streamed text is flushed, so the chat view can keep the newest text in view.
    @Published var scrollTick = 0
    @Published var lastStats: String?
    @Published var systemPrompt: String = "" {
        didSet { saveSystemPrompt() }
    }
    private var stopRequested = false

    /// Hard ceiling for the formatted prompt. LibLlama's batch holds 512 tokens and the context
    /// 1024; staying well below 512 avoids a batch overflow and keeps prompt processing short on
    /// an A13 CPU (the whole remembered history is re-read every turn), leaving ~570+ tokens to reply.
    private let promptTokenBudget = 448
    /// Streamed text is applied to the UI at most this often, so long replies don't re-render per token.
    private let streamFlushInterval: UInt64 = 80_000_000 // ns
    private static let systemPromptKey = "customSystemPrompt"

    @Published var cacheCleared = false
    @Published var downloadedModels: [Model] = []
    @Published var undownloadedModels: [Model] = []
    let NS_PER_S = 1_000_000_000.0

    @Published var serverRunning = false
    @Published var serverAddress = ""
    let httpServer = HTTPServer(port: 8080)

    private var llamaContext: LlamaContext?
    private var defaultModelUrl: URL? {
        Bundle.main.url(forResource: "ggml-model", withExtension: "gguf", subdirectory: "models")
        // Bundle.main.url(forResource: "llama-2-7b-chat", withExtension: "Q2_K.gguf", subdirectory: "models")
    }

    init() {
        if let unfinished = Diagnostics.startSession() {
            previousCrashNote = "The app closed unexpectedly last time during: \(unfinished)"
            messageLog += (previousCrashNote ?? "") + "\n"
        }
        systemPrompt = UserDefaults.standard.string(forKey: Self.systemPromptKey) ?? defaultSystemPrompt
        loadModelsFromDisk()
        loadDefaultModels()
    }

    private func loadModelsFromDisk() {
        do {
            let documentsURL = getDocumentsDirectory()
            let modelURLs = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
            for modelURL in modelURLs {
                let filename = modelURL.lastPathComponent
                let modelName = catalogEntry(forFilename: filename)?.name ?? modelURL.deletingPathExtension().lastPathComponent
                downloadedModels.append(Model(name: modelName, url: "", filename: filename, status: "downloaded"))
            }
        } catch {
            print("Error loading models from disk: \(error)")
        }
    }

    private func loadDefaultModels() {
        do {
            try loadModel(modelUrl: defaultModelUrl)
        } catch {
            messageLog += "Error!\n"
        }

        refreshCatalog()
    }

    /// Catalog entries whose file (under its current or a legacy name) is not on the phone.
    func refreshCatalog() {
        undownloadedModels = defaultModels.filter { model in
            !([model.filename] + model.legacyFilenames).contains { name in
                FileManager.default.fileExists(atPath: getDocumentsDirectory().appendingPathComponent(name).path)
            }
        }
    }

    func catalogEntry(forFilename filename: String) -> Model? {
        defaultModels.first { $0.filename == filename || $0.legacyFilenames.contains(filename) }
    }

    /// Called by the downloaders on the main thread once a file is saved.
    func didDownload(_ model: Model) {
        downloadedModels.removeAll { $0.filename == model.filename }
        downloadedModels.append(model)
        refreshCatalog()
        Diagnostics.mark("downloaded \(model.filename)")
    }

    func didDelete(filenames: [String]) {
        downloadedModels.removeAll { filenames.contains($0.filename) }
        refreshCatalog()
    }

    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
    /// Default download list for the iPhone 11 (4 GB RAM), small to large. Sizes and links from
    /// the keep-survivalpi-phone-models packet (verified 2026-11-18).
    let defaultModels: [Model] = [
        Model(
            name: "LFM2.5 1.2B Instruct",
            url: "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
            filename: "LFM2.5-1.2B-Instruct-Q4_K_M.gguf", status: "download",
            sizeLabel: "0.73 GB", legacyFilenames: ["LFM2"]
        ),
        Model(
            name: "Atlas First Aid 1.5B",
            url: "https://huggingface.co/Kami574/Atlas-FirstAid/resolve/main/Atlas-FirstAid.Q4_K_M.gguf",
            filename: "Atlas-FirstAid.Q4_K_M.gguf", status: "download",
            sizeLabel: "0.99 GB", legacyFilenames: ["Atlas-FirstAid"]
        ),
        Model(
            name: "Qwen2.5 1.5B Instruct",
            url: "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
            filename: "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf", status: "download",
            sizeLabel: "0.99 GB", legacyFilenames: ["Qwen2"]
        ),
        Model(
            name: "SmolLM2 1.7B Instruct",
            url: "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf",
            filename: "smollm2-1.7b-instruct-q4_k_m.gguf", status: "download",
            sizeLabel: "1.06 GB", legacyFilenames: ["smollm2-1"]
        ),
        Model(
            name: "Gemma 2 2B IT",
            url: "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
            filename: "gemma-2-2b-it-Q4_K_M.gguf", status: "download",
            sizeLabel: "1.71 GB", legacyFilenames: ["gemma-2-2b-it-Q4_K_M"]
        ),
        Model(
            name: "Llama 3.2 3B Instruct",
            url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf", status: "download",
            sizeLabel: "2.02 GB", isLarge: true, legacyFilenames: ["Llama-3"]
        ),
    ]

    /// Loads a model off the main thread. The previous model is released first so two models are
    /// never in memory at once (4 GB phone), and taps while a load is running are ignored.
    func loadModel(modelUrl: URL?) throws {
        guard let modelUrl else {
            messageLog += "Load a model from the list below\n"
            return
        }
        guard !isGenerating else {
            messageLog += "Stop the current reply before switching models\n"
            return
        }
        guard !isLoadingModel else { return }

        let filename = modelUrl.lastPathComponent
        let displayName = catalogEntry(forFilename: filename)?.name ?? modelUrl.deletingPathExtension().lastPathComponent
        isLoadingModel = true
        llamaContext = nil
        loadedModelName = nil
        messageLog += "Loading \(displayName)...\n"
        Diagnostics.mark("BEGIN load \(filename)")

        let path = modelUrl.path
        Task {
            do {
                let context = try await Task.detached(priority: .userInitiated) {
                    try LlamaContext.create_context(path: path)
                }.value
                llamaContext = context
                loadedModelName = displayName
                if UserDefaults.standard.string(forKey: Self.systemPromptKey) == nil {
                    systemPrompt = defaultSystemPrompt
                }
                messageLog += "Loaded \(filename)\n"
                Diagnostics.mark("END load \(filename)")
                updateDownloadedModels(modelName: filename, status: "downloaded")
            } catch {
                messageLog += "Could not load \(filename). The file may be incomplete; delete it and download again.\n"
                Diagnostics.mark("END load failed \(filename)")
            }
            isLoadingModel = false
        }
    }

    private func updateDownloadedModels(modelName: String, status: String) {
        undownloadedModels.removeAll { $0.name == modelName }
    }


    // MARK: - Chat

    var hasModel: Bool { llamaContext != nil }

    var defaultSystemPrompt: String {
        if loadedModelName?.lowercased().contains("atlas") == true {
            return """
                You are Atlas, an offline first-aid assistant. Give numbered, step-by-step \
                instructions with the most urgent life-saving steps first (airway and breathing). \
                Be direct and brief. Say when to contact emergency services (911 or 112) if they \
                are reachable. Only give standard first-aid guidance; never invent treatments.
                """
        }
        return "You are a helpful assistant running offline on a phone. Answer clearly and concisely."
    }

    private func saveSystemPrompt() {
        let trimmed = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || systemPrompt == defaultSystemPrompt {
            UserDefaults.standard.removeObject(forKey: Self.systemPromptKey)
        } else {
            UserDefaults.standard.set(systemPrompt, forKey: Self.systemPromptKey)
        }
    }

    func resetSystemPrompt() {
        systemPrompt = defaultSystemPrompt
    }

    func newChat() {
        guard !isGenerating else { return }
        messages.removeAll()
        memoryStartID = nil
    }

    func stopGenerating() {
        stopRequested = true
    }

    func send(_ text: String) {
        let question = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty, !isGenerating, let llamaContext else { return }

        isGenerating = true
        stopRequested = false
        messages.append(ChatMessage(role: .user, text: question))
        let history = messages.filter { !$0.text.isEmpty && !$0.isNotice }
        let reply = ChatMessage(role: .assistant, text: "")
        messages.append(reply)
        scrollTick += 1

        let system = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        Task {
            await generateReply(context: llamaContext, system: system, history: history, replyID: reply.id)
        }
    }

    /// Picks the longest run of recent messages that fits the token budget, always starting
    /// on a user turn and always including the newest question.
    private func buildPrompt(context: LlamaContext, system: String, history: [ChatMessage]) async -> (prompt: String, firstID: UUID)? {
        func format(from start: Int) async -> String {
            var turns: [(role: String, content: String)] = []
            if !system.isEmpty { turns.append((role: "system", content: system)) }
            turns += history[start...].map { (role: $0.role.rawValue, content: $0.text) }
            return await context.format_chat(turns)
        }

        var start = history.count - 1
        var best = await format(from: start)
        guard await context.token_count(text: best) <= promptTokenBudget else { return nil }

        var candidate = start - 2
        while candidate >= 0 {
            if history[candidate].role != .user { candidate -= 1; continue }
            let prompt = await format(from: candidate)
            guard await context.token_count(text: prompt) <= promptTokenBudget else { break }
            start = candidate
            best = prompt
            candidate -= 2
        }
        return (best, history[start].id)
    }

    private func generateReply(context: LlamaContext, system: String, history: [ChatMessage], replyID: UUID) async {
        defer {
            isGenerating = false
            scrollTick += 1
        }

        guard let built = await buildPrompt(context: context, system: system, history: history) else {
            setReply(replyID, "That message is too long for this phone's memory. Please shorten it and try again.")
            if let index = messages.firstIndex(where: { $0.id == replyID }) { messages[index].isNotice = true }
            return
        }
        memoryStartID = built.firstID

        let promptTokens = await context.token_count(text: built.prompt)
        Diagnostics.mark("BEGIN reply (\(promptTokens)-token prompt, \(history.count) messages)")
        let t_start = DispatchTime.now().uptimeNanoseconds
        await context.completion_init(text: built.prompt)
        let t_first = DispatchTime.now().uptimeNanoseconds
        Diagnostics.mark("prompt read in \(String(format: "%.1f", Double(t_first - t_start) / NS_PER_S)) s")

        var reply = ""
        var pending = ""
        var nTokens = 0
        var lastFlush = t_first

        while await !context.is_done, !stopRequested {
            pending += await context.completion_loop()
            nTokens += 1

            let now = DispatchTime.now().uptimeNanoseconds
            if now - lastFlush >= streamFlushInterval {
                reply += pending
                pending = ""
                lastFlush = now
                if Self.trimAtTurnMarker(&reply) { break }
                setReply(replyID, reply.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }
        reply += pending
        _ = Self.trimAtTurnMarker(&reply)
        let t_end = DispatchTime.now().uptimeNanoseconds
        await context.clear()

        var finalText = reply.trimmingCharacters(in: .whitespacesAndNewlines)
        if stopRequested { finalText += finalText.isEmpty ? "(Stopped)" : " …(stopped)" }
        setReply(replyID, finalText)

        let secondsToFirst = Double(t_first - t_start) / NS_PER_S
        let tokensPerSecond = Double(nTokens) / max(Double(t_end - t_first) / NS_PER_S, 0.001)
        lastStats = String(format: "%.1f tok/s, first word after %.1f s", tokensPerSecond, secondsToFirst)
        Diagnostics.mark("END reply (\(nTokens) tokens, \(lastStats ?? ""))")
    }

    /// Cuts the reply at a leaked chat-template marker (the model started a new turn).
    private static func trimAtTurnMarker(_ text: inout String) -> Bool {
        for marker in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "<end_of_turn>", "<start_of_turn>", "<|eot_id|>", "<|start_header_id|>"] {
            if let range = text.range(of: marker) {
                text = String(text[..<range.lowerBound])
                return true
            }
        }
        return false
    }

    private func setReply(_ id: UUID, _ text: String) {
        guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
        messages[index].text = text
        scrollTick += 1
    }

    func conversationText() -> String {
        messages.filter { !$0.text.isEmpty }
            .map { "\($0.role == .user ? "You" : "Assistant"): \($0.text)" }
            .joined(separator: "\n\n")
    }

    func bench() async {
        guard let llamaContext else {
            return
        }

        messageLog += "\n"
        messageLog += "Running benchmark...\n"
        messageLog += "Model info: "
        messageLog += await llamaContext.model_info() + "\n"

        let t_start = DispatchTime.now().uptimeNanoseconds
        let _ = await llamaContext.bench(pp: 8, tg: 4, pl: 1) // heat up
        let t_end = DispatchTime.now().uptimeNanoseconds

        let t_heat = Double(t_end - t_start) / NS_PER_S
        messageLog += "Heat up time: \(t_heat) seconds, please wait...\n"

        // if more than 5 seconds, then we're probably running on a slow device
        if t_heat > 5.0 {
            messageLog += "Heat up time is too long, aborting benchmark\n"
            return
        }

        let result = await llamaContext.bench(pp: 512, tg: 128, pl: 1, nr: 3)

        messageLog += "\(result)"
        messageLog += "\n"
    }

    func clear() async {
        guard let llamaContext else {
            return
        }

        await llamaContext.clear()
        messageLog = ""
    }

    // MARK: - HTTP Server

    func toggleServer() {
        if serverRunning {
            httpServer.stop()
            serverRunning = false
            serverAddress = ""
            messageLog += "Server stopped\n"
        } else {
            do {
                try httpServer.start(llamaState: self)
                serverRunning = true
                let ip = httpServer.getLocalIP()
                serverAddress = "http://\(ip):8080"
                messageLog += "Server started at \(serverAddress)\n"
            } catch {
                messageLog += "Failed to start server: \(error)\n"
            }
        }
    }

    // Non-streaming completion for API use
    func completeForAPI(text: String, maxTokens: Int = 500) async -> String {
        guard let llamaContext else {
            return "Error: No model loaded"
        }

        await llamaContext.completion_init(text: text)
        var result = ""
        var tokenCount = 0

        while await !llamaContext.is_done && tokenCount < maxTokens {
            let piece = await llamaContext.completion_loop()
            result += piece
            tokenCount += 1
        }

        await llamaContext.clear()

        // Strip <think>...</think> blocks from Qwen-style reasoning models
        var cleaned = result
        if let thinkStart = cleaned.range(of: "<think>") {
            if let thinkEnd = cleaned.range(of: "</think>") {
                // Full think block — take everything after it
                cleaned = String(cleaned[thinkEnd.upperBound...])
            } else {
                // Think block consumed all tokens — take everything before it
                cleaned = String(cleaned[..<thinkStart.lowerBound])
            }
        }

        // Stop at first <|im_end|> — model may echo the prompt in a loop
        if let endTag = cleaned.range(of: "<|im_end|>") {
            cleaned = String(cleaned[..<endTag.lowerBound])
        }

        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
