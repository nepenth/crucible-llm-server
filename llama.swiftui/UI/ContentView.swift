import SwiftUI

struct ContentView: View {
    @StateObject var llamaState = LlamaState()
    @State private var draft = ""
    @State private var showingSettings = false

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                ChatTranscript(llamaState: llamaState, onChooseModel: { showingSettings = true })
                Divider()
                ChatInputBar(
                    text: $draft,
                    isGenerating: llamaState.isGenerating,
                    hasModel: llamaState.hasModel,
                    onSend: send,
                    onStop: llamaState.stopGenerating
                )
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    VStack(spacing: 1) {
                        Text("Pocket Assistant")
                            .font(.headline)
                        Text(llamaState.loadedModelName ?? "No model loaded")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button(action: llamaState.newChat) {
                            Label("New Chat", systemImage: "square.and.pencil")
                        }
                        .disabled(llamaState.messages.isEmpty || llamaState.isGenerating)

                        Button(action: { UIPasteboard.general.string = llamaState.conversationText() }) {
                            Label("Copy Conversation", systemImage: "doc.on.doc")
                        }
                        .disabled(llamaState.messages.isEmpty)

                        Button(action: { showingSettings = true }) {
                            Label("Settings", systemImage: "gearshape")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                            .imageScale(.large)
                            .accessibility(label: Text("Menu"))
                    }
                }
            }
            .sheet(isPresented: $showingSettings) {
                SettingsView(llamaState: llamaState)
            }
        }
        .navigationViewStyle(StackNavigationViewStyle())
    }

    private func send() {
        llamaState.send(draft)
        draft = ""
    }
    struct DrawerView: View {

        @ObservedObject var llamaState: LlamaState
        @State private var showingHelp = false
        func delete(at offsets: IndexSet) {
            offsets.forEach { offset in
                let model = llamaState.downloadedModels[offset]
                let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
                do {
                    try FileManager.default.removeItem(at: fileURL)
                } catch {
                    print("Error deleting file: \(error)")
                }
            }

            // Remove models from downloadedModels array
            llamaState.downloadedModels.remove(atOffsets: offsets)
        }

        func getDocumentsDirectory() -> URL {
            let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            return paths[0]
        }
        var body: some View {
            List {
                Section(header: Text("Download Models From Hugging Face")) {
                    HStack {
                        InputButton(llamaState: llamaState)
                    }
                }
                Section(header: Text("Downloaded Models")) {
                    ForEach(llamaState.downloadedModels) { model in
                        DownloadButton(llamaState: llamaState, modelName: model.name, modelUrl: model.url, filename: model.filename)
                    }
                    .onDelete(perform: delete)
                }
                Section(header: Text("Default Models")) {
                    ForEach(llamaState.undownloadedModels) { model in
                        DownloadButton(llamaState: llamaState, modelName: model.name, modelUrl: model.url, filename: model.filename)
                    }
                }

            }
            .listStyle(GroupedListStyle())
            .navigationBarTitle("Model Settings", displayMode: .inline).toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Help") {
                        showingHelp = true
                    }
                }
            }.sheet(isPresented: $showingHelp) {    // Sheet for help modal
                NavigationView {
                    VStack(alignment: .leading) {
                        VStack(alignment: .leading) {
                            Text("1. Make sure the model is in GGUF Format")
                                    .padding()
                            Text("2. Copy the download link of the quantized model")
                                    .padding()
                        }
                        Spacer()
                    }
                    .navigationTitle("Help")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .navigationBarTrailing) {
                            Button("Done") {
                                showingHelp = false
                            }
                        }
                    }
                }
            }
        }
    }
}

private func hideKeyboard() {
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
}

// MARK: - Chat transcript

private struct ChatTranscript: View {
    @ObservedObject var llamaState: LlamaState
    let onChooseModel: () -> Void

    private let bottomID = "bottom"

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                // Plain VStack (not Lazy): conversations are short (bounded by the phone's memory),
                // and LazyVStack's scrollTo is unreliable on iOS 14 while content grows.
                VStack(alignment: .leading, spacing: 0) {
                    if llamaState.messages.isEmpty {
                        EmptyChatView(hasModel: llamaState.hasModel, onChooseModel: onChooseModel)
                    }
                    ForEach(llamaState.messages) { message in
                        if message.id == llamaState.memoryStartID && message.id != llamaState.messages.first?.id {
                            MemoryDivider()
                        }
                        MessageRow(
                            message: message,
                            isPending: llamaState.isGenerating && message.id == llamaState.messages.last?.id && message.text.isEmpty
                        )
                    }
                    Color.clear
                        .frame(height: 8)
                        .id(bottomID)
                }
            }
            .onTapGesture(perform: hideKeyboard)
            .simultaneousGesture(
                DragGesture(minimumDistance: 24).onChanged { value in
                    if value.translation.height > 24 { hideKeyboard() }
                }
            )
            .onChange(of: llamaState.scrollTick) { _ in
                proxy.scrollTo(bottomID, anchor: .bottom)
            }
            .onReceive(NotificationCenter.default.publisher(for: UIResponder.keyboardDidShowNotification)) { _ in
                proxy.scrollTo(bottomID, anchor: .bottom)
            }
        }
    }
}

private struct MessageRow: View {
    let message: ChatMessage
    let isPending: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(message.role == .user ? "You" : "Assistant")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)
            if isPending {
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Thinking…")
                        .font(.body)
                        .foregroundColor(.secondary)
                }
            } else {
                Text(message.text)
                    .font(.body)
                    .foregroundColor(message.isNotice ? .secondary : .primary)
                    .lineSpacing(4)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(message.role == .user ? Color.accentColor.opacity(0.10) : Color.clear)
        .contextMenu {
            Button(action: { UIPasteboard.general.string = message.text }) {
                Label("Copy", systemImage: "doc.on.doc")
            }
        }
    }
}

private struct EmptyChatView: View {
    let hasModel: Bool
    let onChooseModel: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: hasModel ? "bubble.left.and.bubble.right" : "square.and.arrow.down")
                .font(.system(size: 36))
                .foregroundColor(.secondary)
            Text(hasModel ? "Ask a question" : "No model loaded")
                .font(.title3)
                .fontWeight(.semibold)
            Text(hasModel
                 ? "Answers are generated on this phone, with no internet needed. The assistant remembers recent messages in this chat; the oldest are forgotten to fit the phone's memory."
                 : "Download or choose a model to start. Everything runs on this phone, with no internet needed after the download.")
                .font(.callout)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            if !hasModel {
                Button("Choose a Model", action: onChooseModel)
                    .font(.headline)
                    .padding(.top, 4)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, 32)
        .padding(.top, 48)
    }
}

private struct MemoryDivider: View {
    var body: some View {
        HStack(spacing: 8) {
            Rectangle().fill(Color(.separator)).frame(height: 1)
            Text("Older messages are no longer remembered")
                .font(.caption2)
                .foregroundColor(.secondary)
                .layoutPriority(1)
            Rectangle().fill(Color(.separator)).frame(height: 1)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }
}

// MARK: - Input bar

private struct InputHeightKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) { value = max(value, nextValue()) }
}

private struct ChatInputBar: View {
    @Binding var text: String
    let isGenerating: Bool
    let hasModel: Bool
    let onSend: () -> Void
    let onStop: () -> Void

    @State private var textHeight: CGFloat = 0
    private let minHeight: CGFloat = 38
    private let maxHeight: CGFloat = 132 // about five lines at the default text size

    private var canSend: Bool {
        hasModel && !isGenerating && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            ZStack(alignment: .topLeading) {
                // Hidden copy of the text, measured to grow the editor (TextEditor can't self-size on iOS 14).
                Text(text.isEmpty ? " " : text + " ")
                    .font(.body)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 8)
                    .fixedSize(horizontal: false, vertical: true)
                    .foregroundColor(.clear)
                    .background(GeometryReader { geo in
                        Color.clear.preference(key: InputHeightKey.self, value: geo.size.height)
                    })
                TextEditor(text: $text)
                    .font(.body)
                    .disabled(!hasModel)
                if text.isEmpty {
                    Text(hasModel ? "Message" : "Load a model to start")
                        .font(.body)
                        .foregroundColor(Color(.placeholderText))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 8)
                        .allowsHitTesting(false)
                }
            }
            .frame(height: min(max(textHeight, minHeight), maxHeight))
            .clipped()
            .padding(.horizontal, 6)
            .overlay(
                RoundedRectangle(cornerRadius: 19, style: .continuous)
                    .stroke(Color(.separator), lineWidth: 1)
            )
            .onPreferenceChange(InputHeightKey.self) { textHeight = $0 }

            Button(action: isGenerating ? onStop : onSend) {
                Image(systemName: isGenerating ? "stop.circle.fill" : "arrow.up.circle.fill")
                    .font(.system(size: 32))
            }
            .disabled(!isGenerating && !canSend)
            .accessibility(label: Text(isGenerating ? "Stop" : "Send"))
            .padding(.bottom, 3)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(.systemBackground))
    }
}

// MARK: - Settings

private struct SettingsView: View {
    @ObservedObject var llamaState: LlamaState
    @Environment(\.presentationMode) private var presentationMode

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Model")) {
                    HStack {
                        Text("Loaded")
                        Spacer()
                        Text(llamaState.loadedModelName ?? "None")
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    NavigationLink(destination: ContentView.DrawerView(llamaState: llamaState)) {
                        Text("Download or Switch Model")
                    }
                    .disabled(llamaState.isGenerating)
                }

                Section(header: Text("Assistant Instructions"),
                        footer: Text("Sent with every message. Shorter instructions leave more room for the assistant to remember the conversation.")) {
                    TextEditor(text: $llamaState.systemPrompt)
                        .font(.callout)
                        .frame(minHeight: 120)
                    Button("Reset to Default", action: llamaState.resetSystemPrompt)
                        .disabled(llamaState.systemPrompt == llamaState.defaultSystemPrompt)
                }

                Section(header: Text("Local Network Server"),
                        footer: Text("Lets other devices on the same Wi-Fi use this model through an OpenAI-compatible API. Works only while the app is open.")) {
                    Toggle("Server", isOn: Binding(
                        get: { llamaState.serverRunning },
                        set: { _ in llamaState.toggleServer() }
                    ))
                    if !llamaState.serverAddress.isEmpty {
                        Text(llamaState.serverAddress)
                            .font(.footnote)
                            .foregroundColor(.secondary)
                    }
                }

                Section(header: Text("Performance")) {
                    HStack {
                        Text("Last reply")
                        Spacer()
                        Text(llamaState.lastStats ?? "—")
                            .font(.footnote)
                            .foregroundColor(.secondary)
                    }
                    Button("Run Benchmark") {
                        Task { await llamaState.bench() }
                    }
                    .disabled(!llamaState.hasModel || llamaState.isGenerating)
                }

                Section(header: Text("Activity Log")) {
                    Text(llamaState.messageLog.isEmpty ? "No activity yet." : llamaState.messageLog)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Button("Copy Log") { UIPasteboard.general.string = llamaState.messageLog }
                        .disabled(llamaState.messageLog.isEmpty)
                }
            }
            .navigationBarTitle("Settings", displayMode: .inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { presentationMode.wrappedValue.dismiss() }
                }
            }
        }
        .navigationViewStyle(StackNavigationViewStyle())
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
