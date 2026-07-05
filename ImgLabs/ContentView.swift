//
//  ContentView.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-02.
//

import SwiftUI
import PhotosUI

struct AppStatus : View {
    enum Phase {
        case idle
        case importing
        case analyzing
        case finished
        case error
    }
    @State private var curPhase : Phase = .idle;
    @State private var statusMessage : String = "Idle...";
    @State private var progress : Double? = nil;
    @State private var isProgressVisible : Bool = false;
    private let iconSize : CGFloat = 20;
    // Task to reset status to idle
    @State private var resetTask : Task<Void,Never>?; // Task returns nothing and never throws an error
    
    /// Bouncing moon symbol - to indicate no operations are currently ongoing
    private var idleIndicator : some View {
        Image(systemName: "moon.zzz.fill")
            .font(.system(size: self.iconSize))
            .foregroundStyle(.blue)
            .symbolEffect(.bounce.up.byLayer, options: .nonRepeating)
    }
    
    /// Rotating loading symbol - to indicate operation is ongoing
    private var loadingIndicator : some View {
        Image(systemName: "clock.arrow.trianglehead.2.counterclockwise.rotate.90")
            .font(.system(size: self.iconSize))
            .symbolRenderingMode(.hierarchical) // Gives depth to the clock face vs arrows
            .foregroundStyle(.indigo)
            .symbolEffect(
                .rotate.byLayer,
                value: true
            )
    }
    
    /// Bouncing checkmark - to indicate an operation has completed
    private var finishedIndicator : some View {
        Image(systemName: "checkmark.circle.fill")
            .font(.system(size: self.iconSize))
            .foregroundStyle(.green)
            .symbolEffect(.bounce.up.byLayer, options: .nonRepeating)
    }
    
    /// Bouncing error octagon - to indicate an operation failed
    private var errorIndicator : some View {
        Image(systemName: "exclamationmark.octagon.fill")
            .font(.system(size: self.iconSize))
            .foregroundStyle(.red)
            .symbolEffect(.bounce.up.byLayer, options: .nonRepeating)
    }
    
    public func setPhase(to: Phase) {
        if to == .finished { // If finished, then want to return status to idle after sometime
            self.resetTask?.cancel();
            // Start a new async Task
            self.resetTask = Task {
                do {
                    // Sleep for 3 seconds (3,000,000,000 nanoseconds)
                    try await Task.sleep(nanoseconds: 3 * 1_000_000_000);
                    
                    // This runs on the Main Actor automatically to safely update UI
                    await MainActor.run {
                        withAnimation {
                            self.curPhase = .idle; // Reset back to idle
                        }
                    };
                } catch {
                    // The task was canceled before the 3 seconds finished (ignored)
                }
            }
        } else {
            self.resetTask?.cancel();
        }
        withAnimation {
            self.curPhase = to;
        }
    }
    
    public func setStatusMessage(_ msg: String) {
        withAnimation {
            self.statusMessage = msg;
        }
    }
    
    public func setProgress(_ progress: Double) {
        withAnimation {
            self.progress = progress;
        }
    }
    
    @ViewBuilder
    public func getStatusView() -> some View {
        if self.curPhase == .analyzing {
            self.loadingIndicator;
        } else if self.curPhase == .finished {
            self.finishedIndicator;
        } else if self.curPhase == .error {
            self.errorIndicator;
        } else {
            self.idleIndicator;
        }
    }
    
    var body: some View {
        VStack {
            // Status - Icon and Text
            HStack {
                self.getStatusView();
                Text(self.statusMessage)
                    .font(.headline)
                    .fontWeight(.medium)
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: Capsule())
                    .overlay(
                        Capsule().stroke(.white.opacity(0.2), lineWidth: 1)
                    )
            }
            // Add progress bar below status
            ProgressView(value: self.progress, total: 1.0)
                .progressViewStyle(.linear) // Forces the horizontal bar layout
                .tint(self.isProgressVisible ? .white : .gray)
                .animation(.default, value: self.progress)
                .scaleEffect(x: 1, y: 2, anchor: .center) // Make the bar slightly thicker
                .padding(.horizontal, 40)
        }
    }
}

@Observable
class ImageModel {
    private var imageList: [ImageData] = [];
    
    func browseForImages() {
        let openPanel = NSOpenPanel()
        openPanel.title = "Choose an Image"
        openPanel.showsHiddenFiles = false
        openPanel.canChooseDirectories = false
        openPanel.canChooseFiles = true
        openPanel.allowsMultipleSelection = true
        openPanel.allowedContentTypes = [.image] // Filters for images

        // Open the native Finder sheet
        openPanel.begin { response in
            if response == .OK {
                for selectedURL in openPanel.urls { // Multiple images
                    if let nsImage = NSImage(contentsOf: selectedURL),
                       let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                        let newImage = ImageData(img: cgImage);
                        self.imageList.append(newImage);
                    }
                }
            }
        }
    }
    
    func clearImages() {
        self.imageList.removeAll();
    }
    
    func containsImages() -> Bool {
        return !self.imageList.isEmpty;
    }
    
}

struct ControlSideBar : View {
    private let model: ImageModel = ImageModel();
    private let status : AppStatus = AppStatus();
    var body: some View {
        VStack {
            Text("ImgLabs")
                .font(.system(size: 40, weight: .black))
            HStack {
                Button(action: {
                    withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) { // Animate in the Clear button
                        model.browseForImages();
                    }
                    // withAnimation tells swift to interpolate the entire view between two states
                }) {
                    // How the button looks
                    Text("Import Photos")
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .cornerRadius(10)
                };
                if model.containsImages() {
                    Button(action: {
                        withAnimation(.spring(response: 0.4, dampingFraction: 0.6)) {
                            model.clearImages();
                        }
                    }) {
                        // How the button looks
                        Text("Clear Imports")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .cornerRadius(10)
                    }.transition(.move(edge: .trailing).combined(with: .opacity)); // Clear button moves in from right
                }
            }.padding(.horizontal, 20);
            self.status.padding(.vertical, 20);
            Spacer();
        }
        .padding()
        .glassEffect(.regular, in: .rect(cornerRadius: 10)) // Use liquid glass material on the VStack behind the content
        .frame(maxHeight: .infinity)
    }
}

struct ImageGridPane : View {
    var body : some View {
        VStack {
            Text("Image Grid")
                .font(.system(size: 40, weight: .black));
            Spacer();
        }
        .padding()
        .glassEffect(.regular, in: .rect(cornerRadius: 10))
        .frame(maxHeight: .infinity)
    }
}

// View is the fundamental building block of SwiftUI
// It is declarative, i.e., the expected result but supports imperative code
struct ContentView: View { // This a custom view, it conatains a body
    @State private var loadedImageList: [ImageData] = [];
    @State private var isProcessing : Bool = false;
    @State private var numberOfImages: Int = 0;
    
    private let columns = [
        GridItem(.flexible(), spacing: 12),
        GridItem(.flexible(), spacing: 12),
        GridItem(.flexible(), spacing: 12)
    ]
    
    
    func openMacFinder() { // mutating -- method is allowed to change properties of this struct (number of images variable in this case)
        let openPanel = NSOpenPanel()
        openPanel.title = "Choose an Image"
        openPanel.showsHiddenFiles = false
        openPanel.canChooseDirectories = false
        openPanel.canChooseFiles = true
        openPanel.allowsMultipleSelection = true
        openPanel.allowedContentTypes = [.image] // Filters for images

        // Open the native Finder sheet
        openPanel.begin { response in
            if response == .OK {
                // Animate the entrance of the loading bar smoothly
                withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                    self.numberOfImages = openPanel.urls.count
                    self.isProcessing = true
                }
                Task { // Informs swift that items inside this block can be ran on another CPU thread
                    for selectedURL in openPanel.urls { // Multiple images
                        if let nsImage = NSImage(contentsOf: selectedURL),
                           let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                            let newImage = ImageData(img: cgImage);
                            await MainActor.run { self.loadedImageList.append(newImage); } // Context switch back to main ui thread to append, thus forcing view update given the array is a state variable
                            print("Loaded an Image!\n");
                        }
                    }
                    await MainActor.run { // Update main thread UI again to indicate complete
                        withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                            self.isProcessing = false
                        }
                    }
                }
            }
        }
    }
    
    var body: some View { // some means that body returns an object which conforms to the View type (i.e., don't need to specify exactly what is returned)
        HStack {
            ImageGridPane()
            ControlSideBar()
        }.glassEffect(.regular, in: RoundedRectangle(cornerRadius: 0, style: .continuous));
    }
}

#Preview {
    ContentView()
}

// @state tells SwiftUI to watch the value for any changes, if so body will be called again
// @binding allows another view to watch a different view's @state
// Thus it too will trigger body if the @state from another view changes
