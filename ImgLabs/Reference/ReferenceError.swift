//
//  ReferenceError.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-20
//  Denotes all throwable errors related to references

import Foundation

public enum ReferenceError: Error {
    case invalidLibraryDirectory;
    case invalidReferenceLibraryFile(String);
    case invalidAppSupportDirectory;
}

