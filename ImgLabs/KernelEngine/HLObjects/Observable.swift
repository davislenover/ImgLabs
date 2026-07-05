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


/// A thread-safe class which stores and calls observers update functions
/// Useful as an Observer store for ObservableResult objects
public actor ObserverStore {
    private var funcs : [(Any) async -> ()] = []; // Store update functions instead to allow for passing of any type value (the type will be checked within the function)
    
    public func add<O: ResultObserver>(_ observer: O) {
        let updateFunc : @Sendable (Any) async -> () = { value in
            guard let typedValue = value as? O.Result else {print(O.Result.self, value.self);return;} // Result is known at compile time
            await observer.update(with: typedValue);
        }
        funcs.append(updateFunc);
    }
    
    public func callAll(with: Any) async {
        for f in funcs {
            await f(with);
        }
    }
}

