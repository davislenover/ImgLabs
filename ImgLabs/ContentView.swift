//
//  ContentView.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-02.
//  The app's root view: composes the results grid and the control sidebar over shared app state

import SwiftUI
import UniformTypeIdentifiers

public let PROGRESS_END : CGFloat = 1.0;
public let PROGRESS_START : CGFloat = 0.0;

// View is the fundamental building block of SwiftUI
// It is declarative, i.e., the expected result but supports imperative code
struct ContentView: View { // This a custom view, it conatains a body
    // The shared app state lives here (the common parent) so both panes can read it: the sidebar drives
    // imports/analysis, and the grid pane displays the results. @State keeps the instances alive across
    // re-renders; being @Observable classes, passing them to child views lets those views observe changes
    @State private var model : ImageModel = ImageModel();
    @State private var status : AppStatusModel = AppStatusModel();
    @State private var znccObj : ZNCCModel = ZNCCModel();
    @State private var maxCanvasDim : Double = 512;
    @State private var isDraggingOver : Bool = false;

    private func importDroppedURLs(_ urls: [URL]) {
        guard !status.isBusy else { return; }
        withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
            model.importImages(from: urls, status, maxCanvasDim: Int(maxCanvasDim));
        }
    }

    private func loadFileURL(from provider: NSItemProvider, completion: @escaping (URL?) -> Void) {
        let identifiers = provider.registeredTypeIdentifiers;
        print("SwiftUI drop provider types: \(identifiers)");

        let candidateIdentifiers = ([
            UTType.fileURL.identifier,
            UTType.folder.identifier,
            UTType.image.identifier,
            UTType.item.identifier
        ] + identifiers).uniqued();

        func decodedURL(from item: NSSecureCoding?) -> URL? {
            if let url = item as? URL {
                return url;
            }
            if let url = item as? NSURL {
                return url as URL;
            }
            if let data = item as? Data {
                if let url = URL(dataRepresentation: data, relativeTo: nil) {
                    return url;
                }
                if let string = String(data: data, encoding: .utf8) {
                    return URL(string: string) ?? URL(fileURLWithPath: string);
                }
            }
            if let string = item as? String {
                return URL(string: string) ?? URL(fileURLWithPath: string);
            }
            if let string = item as? NSString {
                let value = string as String;
                return URL(string: value) ?? URL(fileURLWithPath: value);
            }
            return nil;
        }

        func tryLoadingItem(at index: Int) {
            guard index < candidateIdentifiers.count else {
                tryLoadingFileRepresentation(at: 0);
                return;
            }

            let identifier = candidateIdentifiers[index];
            guard provider.hasItemConformingToTypeIdentifier(identifier) else {
                tryLoadingItem(at: index + 1);
                return;
            }

            provider.loadItem(forTypeIdentifier: identifier, options: nil) { item, _ in
                if let url = decodedURL(from: item) {
                    completion(url);
                } else {
                    tryLoadingItem(at: index + 1);
                }
            }
        }

        func tryLoadingFileRepresentation(at index: Int) {
            guard index < candidateIdentifiers.count else {
                completion(nil);
                return;
            }

            let identifier = candidateIdentifiers[index];
            guard provider.hasItemConformingToTypeIdentifier(identifier) else {
                tryLoadingFileRepresentation(at: index + 1);
                return;
            }

            _ = provider.loadInPlaceFileRepresentation(forTypeIdentifier: identifier) { url, _, _ in
                if let url {
                    completion(url);
                } else {
                    tryLoadingFileRepresentation(at: index + 1);
                }
            }
        }

        tryLoadingItem(at: 0);
    }

    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        print("SwiftUI dropped \(providers.count) item provider(s)");
        guard !status.isBusy else { return false; }

        let group = DispatchGroup();
        let lock = NSLock();
        var urls: [URL] = [];
        for provider in providers {
            group.enter();
            loadFileURL(from: provider) { url in
                if let url {
                    lock.lock();
                    urls.append(url);
                    lock.unlock();
                }
                group.leave();
            }
        }

        group.notify(queue: .main) {
            guard !urls.isEmpty else {
                print("SwiftUI drop did not contain readable file URLs");
                return;
            }
            importDroppedURLs(urls);
        }
        return true;
    }

    var body: some View { // some means that body returns an object which conforms to the View type (i.e., don't need to specify exactly what is returned)
        HStack {
            ImageGridPane(model: self.model, status: self.status, znccObj: self.znccObj)
            ControlSideBar(model: self.model, status: self.status, znccObj: self.znccObj, maxCanvasDim: $maxCanvasDim)
        }
        .glassEffect(.regular, in: RoundedRectangle(cornerRadius: 0, style: .continuous))
        .contentShape(Rectangle())
        .onDrop(of: [.fileURL, .folder, .image], isTargeted: $isDraggingOver) { providers in
            handleDrop(providers: providers);
        }
        .overlay {
            if isDraggingOver {
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 3, dash: [8]))
                    .padding(8)
                    .allowsHitTesting(false);
            }
        };
    }
}

#Preview {
    ContentView()
}

private extension Array where Element: Hashable {
    func uniqued() -> [Element] {
        var seen = Set<Element>();
        return filter { seen.insert($0).inserted };
    }
}

// @state tells SwiftUI to watch the value for any changes, if so body will be called again
// @binding allows another view to watch a different view's @state
// Thus it too will trigger body if the @state from another view changes
