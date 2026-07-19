//
//  AppStatus.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-03-02.
//  The app's status indicator: an observable phase/message/progress model and the view that renders it

import SwiftUI

// Holds the state for the app's status indicator
// This is a reference type (@Observable class), NOT a View, so that mutations-
// made from anywhere (e.g. ImageModel) are seen by the View that renders it
@Observable
class AppStatusModel {
    enum Phase {
        case idle
        case importing
        case analyzing
        case clearing
        case browsing
        case copying
        case benchmarking
        case finished
        case error
    }
    var curPhase : Phase = .idle;
    var statusMessage : String = "Idle...";
    var progress : Double? = nil;
    var isProgressVisible : Bool = false;
    // Task to reset status to idle
    @ObservationIgnored private var resetTask : Task<Void,Never>?; // Task returns nothing and never throws an error

    /// True while an operation is in progress - used to disable controls & drive the spinner
    var isBusy : Bool {
        return self.curPhase == .importing || self.curPhase == .analyzing || self.curPhase == .clearing || self.curPhase == .browsing || self.curPhase == .copying || self.curPhase == .benchmarking;
    }

    func setPhase(to: Phase) {
        if to == .finished || to == .error { // If finished, then want to return status to idle after sometime
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
                            self.isProgressVisible = false;
                            self.progress = nil;
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

    func setStatusMessage(_ msg: String) {
        withAnimation {
            self.statusMessage = msg;
        }
    }

    func setProgress(_ progress: Double) {
        withAnimation {
            if (!self.isProgressVisible) {
                self.isProgressVisible = true;
            }
            self.progress = progress;
        }
    }
}

// Renders the status indicator described by an AppStatusModel
struct AppStatus : View {
    let model : AppStatusModel;
    private let iconSize : CGFloat = 20;

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
                options: .repeat(.continuous), // Spin smoothly & continuously (not a single discrete turn)
                isActive: self.model.isBusy
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

    @ViewBuilder
    public func getStatusView() -> some View {
        if self.model.isBusy {
            self.loadingIndicator;
        } else if self.model.curPhase == .finished {
            self.finishedIndicator;
        } else if self.model.curPhase == .error {
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
                Text(self.model.statusMessage)
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
            // Always a determinate bar: when there's no progress (idle / after finished) it sits at 0 as a
            // static, empty track instead of an animated indeterminate sweep. progress ?? 0 supplies the
            // fallback value when model.progress is nil
            ProgressView(value: self.model.progress ?? 0, total: PROGRESS_END)
            .progressViewStyle(.linear) // Forces the horizontal bar layout
            .tint(self.model.isProgressVisible ? .white : .gray)
            .animation(.default, value: self.model.progress)
            .scaleEffect(x: 1, y: 2, anchor: .center) // Make the bar slightly thicker
            .padding(.horizontal, 40)
        }
    }
}
