//
//  ContentView.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-02.
//

import SwiftUI
import PhotosUI

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
                            let newImage = await ImageData(img: cgImage); // Spawned on another thread
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
        Text("ImgLabs").font(.largeTitle.bold()).padding()
        VStack {
            if !self.isProcessing && self.loadedImageList.isEmpty {
                ZStack { // Z direction stack (i.e., stack items on top of eachother)
                    RoundedRectangle(cornerRadius: 5)
                        .fill(Color.gray).padding(10)
                
                    Button("Import Image") {
                        openMacFinder()
                    }
                    .shadow(color: .black.opacity(0.7), radius: 20)
                    .shadow(color: .black.opacity(0.7), radius: 20)
                }.transition(.slide)
            } else if self.isProcessing {
                ZStack { // Z direction stack (i.e., stack items on top of eachother)
                    RoundedRectangle(cornerRadius: 5)
                        .fill(Color.clear).padding(10)
                
                    // Show progress bar with images that are loading
                    ProgressView("Loaded \(self.loadedImageList.count) out of \(self.numberOfImages)", value: Float(self.loadedImageList.count), total: Float(self.numberOfImages)).padding(25)
                }
            } else {
                // Loaded image list is not empty, display list
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 12) {
                        ForEach(loadedImageList) { item in
                            // Render the thumbnail (Convert CGImage/NSImage to SwiftUI Image)
                            Image(nsImage: NSImage(cgImage: item.getCGImage()!, size: .zero))
                                .resizable()
                                .scaledToFit()
                                .frame(minWidth: 0, maxWidth: .infinity, minHeight: 120, maxHeight: 120)
                                .clipped()
                                .cornerRadius(8)
                                .shadow(radius: 2)
                        }
                    }.padding()
                }.transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }.animation(.spring(response: 0.4, dampingFraction: 0.8), value: self.isProcessing)
    }
}

#Preview {
    ContentView()
}

// @state tells SwiftUI to watch the value for any changes, if so body will be called again
// @binding allows another view to watch a different view's @state
// Thus it too will trigger body if the @state from another view changes
