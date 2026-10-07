//
//  Theme.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-19.
//  The app's colour palette and the shared button style built on it

import SwiftUI

extension Color {
    /// Build a Color from a 0xRRGGBB literal (sRGB, fully opaque)
    init(hex: UInt32) {
        self.init(.sRGB,
                  red: Double((hex >> 16) & 0xFF) / 255,
                  green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255,
                  opacity: 1);
    }

    // MARK: - Palette
    /// Deep blue — primary actions and the app's main accent
    static let brandPrimary = Color(hex: 0x006BBB);
    /// Bright blue — secondary actions and control tints
    static let brandSecondary = Color(hex: 0x30A0E0);
    /// Warm amber — the hero call-to-action and highlights
    static let brandAccent = Color(hex: 0xFFC872);
    /// Pale amber — subtle fills behind captions / hints
    static let brandAccentLight = Color(hex: 0xFFE3B3);
}

// Exposes the palette to the leading-dot shorthand in ShapeStyle contexts (foregroundStyle, tint, fill,
// stroke, ...), the same way SwiftUI's built-in `.red` etc. work. Without this, `.brandPrimary` only
// resolves where a `Color` is expected directly (e.g. a `Color` parameter or `.background(_:)`)
extension ShapeStyle where Self == Color {
    static var brandPrimary: Color { Color.brandPrimary }
    static var brandSecondary: Color { Color.brandSecondary }
    static var brandAccent: Color { Color.brandAccent }
    static var brandAccentLight: Color { Color.brandAccentLight }
}

/// A filled, full-width action button used across the app's controls. `fill` sets the background and
/// `foreground` the label colour. It dims when disabled and darkens slightly while pressed, so every button
/// reads as an obvious, consistent control
struct FilledActionButtonStyle: ButtonStyle {
    var fill: Color;
    var foreground: Color = .white;

    func makeBody(configuration: Configuration) -> some View {
        StyledLabel(configuration: configuration, fill: fill, foreground: foreground);
    }

    // A nested View so the style can read the enclosing button's enabled state from the environment
    private struct StyledLabel: View {
        let configuration: ButtonStyleConfiguration;
        let fill: Color;
        let foreground: Color;
        @Environment(\.isEnabled) private var isEnabled;

        var body: some View {
            configuration.label
                .font(.headline)
                .foregroundStyle(isEnabled ? foreground : .white.opacity(0.8))
                // Keep every button one line tall so a row of buttons stays a uniform height; long labels
                // shrink only slightly (down to 90%) in narrow columns instead of wrapping onto a second line
                .lineLimit(1)
                .minimumScaleFactor(0.9)
                .padding(.vertical, 10)
                .padding(.horizontal, 14)
                .frame(maxWidth: .infinity)
                .background((isEnabled ? fill : Color.gray)
                    .opacity(configuration.isPressed ? 0.7 : 1),
                            in: RoundedRectangle(cornerRadius: 10))
                .opacity(isEnabled ? 1 : 0.5)
                .contentShape(RoundedRectangle(cornerRadius: 10))
                .animation(.easeOut(duration: 0.12), value: configuration.isPressed);
        }
    }
}
