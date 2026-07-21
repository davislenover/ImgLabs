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
    public func getReferenceLibrary(_ id: UUID) throws -> ReferenceLibrary? {
        if try self.doesReferenceLibraryExist(id) {
            guard let libraryURL : URL  = self.appSupportDirectory?.appendingPathComponent(ReferenceDirectory.REFERENCES_LIBRARIES.rawValue).appendingPathComponent(id.uuidString).appendingPathExtension("json") else {
                return nil;
            }
            let decoder = JSONDecoder();
            return try decoder.decode(ReferenceLibrary.self, from: try Data(contentsOf: libraryURL));
        } else {
            return nil;
        }
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
    
    /// Re-scans a stored library against the current state of disk and persists any changes.
    /// Loads the library fresh from disk, re-hashes only new/modified files via the indexer, and commits the
    /// result only when something actually changed
    /// - Parameters:
    ///     - id: the library to re-scan
    ///     - indexer: builds/re-hashes reference entries (owns the Metal context)
    /// - Returns: the updated library when changes were written, or nil when nothing changed
    /// - Throws: `referenceLibraryNotFound` if no library with that id is stored
    public func rescanReferenceLibrary(_ id: UUID, _ indexer: ReferenceIndexer) async throws -> ReferenceLibrary? {
        guard let library : ReferenceLibrary = try self.getReferenceLibrary(id) else {
            throw ReferenceError.referenceLibraryNotFound(id);
        }
        // rescan returns nil when nothing on disk changed, so there's nothing to persist
        guard let updated : ReferenceLibrary = try await indexer.rescan(library) else {
            return nil;
        }
        try self.commitReferenceLibrary(updated);
        return updated;
    }

    /// Commit changes to a given ReferenceLibrary
    public func commitReferenceLibrary(_ library: ReferenceLibrary) throws {
        let encoder : JSONEncoder = JSONEncoder();
        encoder.outputFormatting = .prettyPrinted;
        let data : Data = try encoder.encode(library);
        let libraryURL : URL = self.appSupportDirectory!.appendingPathComponent(ReferenceDirectory.REFERENCES_LIBRARIES.rawValue).appendingPathComponent(library.id.uuidString).appendingPathExtension("json");
        try data.write(to: libraryURL);
    }
    
    /// Checks each candidate pHash for a match in a given ReferenceLibrary (within a Hamming tolerance)
    /// - Parameters:
    ///     - pHashes: the candidate image hashes to look up
    ///     - library: the reference library to match against
    ///     - bitCountTolarance: max differing bits for a pair to count as a match
    /// - Returns: a list index-aligned with `pHashes`, each element the first library entry within tolerance
    ///            of that candidate (nil when none match)
    public static func checkForReference(_ pHashes : [ImageHash], _ library: ReferenceLibrary, _ bitCountTolarance: UInt8) async -> [ReferenceEntry?] {
        let entries : [ReferenceEntry] = library.entries;
        let tolerance : Int = Int(bitCountTolarance);

        // Pull each candidate's 64-bit hash off its actor exactly once (not once per library entry)
        var candidateValues : [UInt64] = [];
        candidateValues.reserveCapacity(pHashes.count);
        for pHash in pHashes {
            candidateValues.append(await pHash.value());
        }

        // Each candidate scans the library independently so fan out over candidates and collect results back into their original slots
        var results : [ReferenceEntry?] = Array<ReferenceEntry?>(repeating: nil, count: pHashes.count);
        await withTaskGroup(of: (Int, ReferenceEntry?).self) { group in
            for (index, value) in candidateValues.enumerated() {
                group.addTask {
                    for entry in entries {
                        if (entry.hash ^ value).nonzeroBitCount <= tolerance {
                            return (index, entry); // first entry within tolerance for this candidate
                        }
                    }
                    return (index, nil);
                }
            }
            for await (index, match) in group {
                results[index] = match;
            }
        }
        return results;
    }
}
