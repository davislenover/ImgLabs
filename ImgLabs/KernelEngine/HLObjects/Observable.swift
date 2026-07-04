//
//  Observable.swift
//  ImgLabs
//
//  Created by Davis Lenover on 2026-07-03.
//  Denotes a protcol following the observer pattern

import Foundation

/// Implementing objects can be observed for some result (e.g., a Kernel finishing)
public protocol ObservableResult {
    /// Adds a ResultObserver to the list of observers or rather gets the update function to call when notifyObservers() is called
    /// - Parameters:
    ///     - observer: An object of ResultObserver<Result> type. If the Result type is not what is returned via the protocol implementing object, update() is never called for the given observer
    func addObserver<O: ResultObserver>(_ observer: O) async;
    
    /// Notifies all observers who have subscribed via addObserver
    func notifyObservers() async;
}

/// Implementing objects can observe objects which implement ObservableResult
public protocol ResultObserver<Result> {
    associatedtype Result;
    /// Called if notifyObservers() was invoked from a ObservableResult object subscribed to
    /// Only invoked it the type of Result matches was ObservableResult can provide
    func update(with: Result) async;
}

