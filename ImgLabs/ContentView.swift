//
//  ContentView.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-02.
//

import SwiftUI
import PhotosUI


struct ControlSideBar : View {
    
    var body: some View {
        VStack {
            Text("ImgLabs")
                .font(.system(size: 40, weight: .black))
            Button(action: {
                // TODO: add file open
                return;
            }) {
                // How the button looks
                Text("Import Photos")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .cornerRadius(10)
            }
            .padding(.horizontal, 40);
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
