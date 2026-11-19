import Foundation

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
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
        systemPrompt = UserDefaults.standard.string(forKey: Self.systemPromptKey) ?? defaultSystemPrompt
        loadModelsFromDisk()
        loadDefaultModels()
    }

    private func loadModelsFromDisk() {
        do {
            let documentsURL = getDocumentsDirectory()
            let modelURLs = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
            for modelURL in modelURLs {
                let modelName = modelURL.deletingPathExtension().lastPathComponent
                downloadedModels.append(Model(name: modelName, url: "", filename: modelURL.lastPathComponent, status: "downloaded"))
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

        for model in defaultModels {
            let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
            if FileManager.default.fileExists(atPath: fileURL.path) {

            } else {
                var undownloadedModel = model
                undownloadedModel.status = "download"
                undownloadedModels.append(undownloadedModel)
            }
        }
    }

    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
    private let defaultModels: [Model] = [
        Model(
            name: "Gemma-3-4B-IT (Q4_K_M, 2.9 GiB)",
            url: "https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF/resolve/main/google_gemma-3-4b-it-Q4_K_M.gguf?download=true",
            filename: "gemma-3-4b-it-Q4_K_M.gguf", status: "download"
        ),
        Model(
            name: "Qwen-3.5-4B (Q4_K_M, 2.7 GiB)",
            url: "https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF/resolve/main/Qwen_Qwen3.5-4B-Q4_K_M.gguf?download=true",
            filename: "qwen3.5-4b-Q4_K_M.gguf", status: "download"
        ),
        Model(name: "TinyLlama-1.1B (Q4_0, 0.6 GiB)",url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_0.gguf?download=true",filename: "tinyllama-1.1b-1t-openorca.Q4_0.gguf", status: "download"),
        Model(
            name: "TinyLlama-1.1B Chat (Q8_0, 1.1 GiB)",
            url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true",
            filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf", status: "download"
        ),

        Model(
            name: "TinyLlama-1.1B (F16, 2.2 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-f16.gguf?download=true",
            filename: "tinyllama-1.1b-f16.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q4_0, 1.6 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q4_0.gguf?download=true",
            filename: "phi-2-q4_0.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q8_0, 2.8 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q8_0.gguf?download=true",
            filename: "phi-2-q8_0.gguf", status: "download"
        ),

        Model(
            name: "Mistral-7B-v0.1 (Q4_0, 3.8 GiB)",
            url: "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_0.gguf?download=true",
            filename: "mistral-7b-v0.1.Q4_0.gguf", status: "download"
        ),
        Model(
            name: "OpenHermes-2.5-Mistral-7B (Q3_K_M, 3.52 GiB)",
            url: "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q3_K_M.gguf?download=true",
            filename: "openhermes-2.5-mistral-7b.Q3_K_M.gguf", status: "download"
        )
    ]
    func loadModel(modelUrl: URL?) throws {
        if let modelUrl {
            guard !isGenerating else {
                messageLog += "Stop the current reply before switching models\n"
                return
            }
            messageLog += "Loading model...\n"
            llamaContext = try LlamaContext.create_context(path: modelUrl.path)
            loadedModelName = modelUrl.deletingPathExtension().lastPathComponent
            if UserDefaults.standard.string(forKey: Self.systemPromptKey) == nil {
                systemPrompt = defaultSystemPrompt
            }
            messageLog += "Loaded model \(modelUrl.lastPathComponent)\n"

            // Assuming that the model is successfully loaded, update the downloaded models
            updateDownloadedModels(modelName: modelUrl.lastPathComponent, status: "downloaded")
        } else {
            messageLog += "Load a model from the list below\n"
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

        let t_start = DispatchTime.now().uptimeNanoseconds
        await context.completion_init(text: built.prompt)
        let t_first = DispatchTime.now().uptimeNanoseconds

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
    }

    /// Cuts the reply at a leaked chat-template marker (the model started a new turn).
    private static func trimAtTurnMarker(_ text: inout String) -> Bool {
        for marker in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "<end_of_turn>"] {
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
