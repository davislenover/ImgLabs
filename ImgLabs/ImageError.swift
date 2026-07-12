//
//  ImageError.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-02.
//


enum ImageError: Error {
    case noPixelData;
    case failedToConvertDataToMTLBuffer;
    case failedToResizeImage;
}
