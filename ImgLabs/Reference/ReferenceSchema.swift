//
//  ReferenceSchema.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-20
//  Describes the model used to store file references for lookup when comparing images to reference libraries
//  A single file should make up a ReferenceLibrary

import Foundation

/// Denotes a directory that is apart of a ReferenceLibrary, Codeable to allow for serialization
struct LibraryRoot : Codable {
    let id: UUID;
    let bookmark: Data; // security-scoped bookmark to the given directory
    let displayPath: String; // for the UI (is the file path)
}

/// Denotes a single image within a LibraryRoot (i.e., a directory contains multiple images, thus a LibraryRoot contains ReferenceEntries)
/// Houses information about the image, such as it's computed pHash and name
struct ReferenceEntry: Codable {
    let hash: UInt64;
    let libraryRootID: UUID; // which root this file lives under
    let relativePath: String; // path within that root
    let fileName: String;
    let byteSize: Int64;
    let modified: Date;
}

/// Denotes a group of LibraryRoots (allows the user to combine library roots into one reference library)
struct ReferenceLibrary: Codable {
    let id: UUID;
    var name: String; // User adjustable
    var roots: [LibraryRoot];
    var entries: [ReferenceEntry];
    var lastScanned: Date?;
    var formatVersion: Int = 1;
}
