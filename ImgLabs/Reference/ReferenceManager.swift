//
//  ReferenceManager.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-20
//  An actor which manages reference data to/from the application

import Foundation

public actor ReferenceManager {
    private let fileManager: FileManager = .default;
    private var appSupportDirectory: URL?;
    
    // Directories within support directory
    private enum ReferenceDirectory : String, CaseIterable {
        case REFERENCES_LIBRARIES = "libReferences";
    }
    
    /// Gets the main application data directory, callers will need to append to this path to look for specific items
    /// Handles creation of folder structure
    private static func getAppSupportDirectory(_ manager : FileManager) throws -> URL {
        let appURL : URL = try manager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true // Automatically creates the folder if it does not exist
        );
        // Check if each directory within the ReferenceDirectory has been created, otherwise create it
        for directory in ReferenceDirectory.allCases {
            let directoryURL : URL = appURL.appendingPathComponent(directory.rawValue); // .rawValue gets String
            if !manager.fileExists(atPath: directoryURL.path) {
                try manager.createDirectory(at: directoryURL, withIntermediateDirectories: true, attributes: nil);
            }
        }
        return appURL;
    }
    
    init() {
        // Create support directory if it doesn't exist
        self.appSupportDirectory = try? ReferenceManager.getAppSupportDirectory(self.fileManager);
    }
    
    /// Gets all ReferenceLibrary objects stored
    public func getAllReferenceLibraries() throws -> [ReferenceLibrary] {
        guard let mainDir = self.appSupportDirectory else {
            throw ReferenceError.invalidAppSupportDirectory;
        }
        let libraryDirectories : [URL] = try self.fileManager.contentsOfDirectory(at: mainDir.appendingPathComponent(ReferenceDirectory.REFERENCES_LIBRARIES.rawValue), includingPropertiesForKeys: nil, options: .skipsHiddenFiles);
        // For each URL, pull data from the file, attempt conversion to a ReferenceLibrary then return all
        var referenceLibraries : [ReferenceLibrary] = [];
        for libraryURL in libraryDirectories {
            guard let libraryData : Data = try? Data(contentsOf: libraryURL), let library : ReferenceLibrary = try? JSONDecoder().decode(ReferenceLibrary.self, from: libraryData) else {
                throw ReferenceError.invalidReferenceLibraryFile(libraryURL.path());
            }
            referenceLibraries.append(library);
        }
        return referenceLibraries;
    }
    
    /// Gets a specific ReferenceLibraryObject from it's UUID
    public func getReferenceLibrary(_ id: UUID) -> ReferenceLibrary? {
        let libraries : [ReferenceLibrary] = (try? self.getAllReferenceLibraries()) ?? [];
        return libraries.first(where: { $0.id == id });
    }
    
    /// Checks if a file name within the reference library directory matches a given id
    public func doesReferenceLibraryExist(_ id: UUID) throws -> Bool {
        guard let mainDir = self.appSupportDirectory else {
            throw ReferenceError.invalidAppSupportDirectory;
        }
        guard let libraryDirectories : [URL] = try? self.fileManager.contentsOfDirectory(at: mainDir.appendingPathComponent(ReferenceDirectory.REFERENCES_LIBRARIES.rawValue), includingPropertiesForKeys: nil, options: .skipsHiddenFiles) else {
            throw ReferenceError.invalidAppSupportDirectory;
        }
        for libraryURL in libraryDirectories {
            if libraryURL.lastPathComponent.replacingOccurrences(of: ".json", with: "") == id.uuidString {
                return true;
            }
        }
        return false;
    }
    
    /// Creates a ReferenceLibrary object and writes it's contents to application data
    public func createReferenceLibrary(_ name : String, _ roots : [LibraryRoot], _ ReferenceEntries : [ReferenceEntry]) throws -> ReferenceLibrary {
        var id : UUID = UUID();
        while true {
            // Check if reference library with UUID exists (near-zero chance)
            if try !self.doesReferenceLibraryExist(id) {
                break;
            }
            id = UUID();
        }
        let library : ReferenceLibrary = ReferenceLibrary(id: id, name: name, roots: roots, entries: ReferenceEntries, lastScanned: Date());
        // Store library
        let encoder : JSONEncoder = JSONEncoder();
        encoder.outputFormatting = .prettyPrinted;
        let data : Data = try encoder.encode(library);
        // Create file URL (uuid of library)
        let libraryURL : URL = self.appSupportDirectory!.appendingPathComponent(ReferenceDirectory.REFERENCES_LIBRARIES.rawValue).appendingPathComponent(id.uuidString).appendingPathExtension("json");
        // Write data to file
        try data.write(to: libraryURL);
        return library;
    }
    
}
