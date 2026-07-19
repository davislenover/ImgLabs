//
//  ContentView.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-02.
//  The app's root view: composes the results grid and the control sidebar over shared app state

import SwiftUI

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

    var body: some View { // some means that body returns an object which conforms to the View type (i.e., don't need to specify exactly what is returned)
        HStack {
            ImageGridPane(model: self.model, status: self.status, znccObj: self.znccObj)
            ControlSideBar(model: self.model, status: self.status, znccObj: self.znccObj)
        }.glassEffect(.regular, in: RoundedRectangle(cornerRadius: 0, style: .continuous));
    }
}

#Preview {
    ContentView()
}

// @state tells SwiftUI to watch the value for any changes, if so body will be called again
// @binding allows another view to watch a different view's @state
// Thus it too will trigger body if the @state from another view changes
