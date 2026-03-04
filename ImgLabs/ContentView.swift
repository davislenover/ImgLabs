//
//  ContentView.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-02.
//

import SwiftUI

// View is the fundamental building block of SwiftUI
// It is declarative, i.e., the expected result but supports imperative code
struct ContentView: View { // This a custom view, it conatains a body
    
    func importImage() -> Void {
        
    }
    
    var body: some View {
        Text("ImgLabs").font(.largeTitle.bold()).padding()
        ZStack { // Z direction stack (i.e., stack items on top of eachother)
            RoundedRectangle(cornerRadius: 5)
                .fill(Color.gray).padding(10)
            
            Button(role: ButtonRole.confirm, action: importImage) {
                Label("Import Image", systemImage:"square.and.arrow.down.fill")
            }
                .shadow(color: .black.opacity(0.7), radius: 20)
                .shadow(color: .black.opacity(0.7), radius: 20)
        }
    }
}

#Preview {
    ContentView()
}

// @state tells SwiftUI to watch the value for any changes, if so body will be called again
// @binding allows another view to watch a different view's @state
// Thus it too will trigger body if the @state from another view changes
