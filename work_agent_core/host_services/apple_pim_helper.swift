import EventKit
import Foundation

typealias JsonObject = [String: Any]

struct ApplePimError: Error {
    let code: String
    let message: String
}

func jsonData(_ value: Any) throws -> Data {
    try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
}

func printResponse(_ value: JsonObject) {
    guard let data = try? jsonData(value), let text = String(data: data, encoding: .utf8) else {
        print("{\"ok\":false,\"error\":{\"code\":\"SERIALIZATION_FAILED\",\"message\":\"Unable to serialize response.\"}}")
        return
    }
    print(text)
}

func stringValue(_ object: JsonObject, _ key: String, required: Bool = false) throws -> String {
    let value = (object[key] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    if required && value.isEmpty {
        throw ApplePimError(code: "INVALID_INPUT", message: "Missing required field: \(key).")
    }
    return value
}

func boolValue(_ object: JsonObject, _ key: String, fallback: Bool = false) -> Bool {
    object[key] as? Bool ?? fallback
}

func isoFormatter() -> ISO8601DateFormatter {
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return formatter
}

func parseDate(_ raw: String, field: String, required: Bool = false) throws -> Date? {
    if raw.isEmpty {
        if required {
            throw ApplePimError(code: "INVALID_INPUT", message: "Missing required date: \(field).")
        }
        return nil
    }
    let formatter = isoFormatter()
    if let date = formatter.date(from: raw) {
        return date
    }
    let fallback = ISO8601DateFormatter()
    fallback.formatOptions = [.withInternetDateTime]
    if let date = fallback.date(from: raw) {
        return date
    }
    throw ApplePimError(code: "INVALID_INPUT", message: "\(field) must be an ISO 8601 timestamp.")
}

func isoString(_ date: Date?) -> String {
    guard let date else { return "" }
    return isoFormatter().string(from: date)
}

func boundedString(_ value: String?, maximumLength: Int) -> String {
    let text = value ?? ""
    guard text.count > maximumLength else { return text }
    return String(text.prefix(maximumLength)) + "…"
}

func authorizationName(_ status: EKAuthorizationStatus) -> String {
    if #available(macOS 14.0, *) {
        switch status {
        case .fullAccess:
            return "full_access"
        case .writeOnly:
            return "write_only"
        case .notDetermined:
            return "not_determined"
        case .denied:
            return "denied"
        case .restricted:
            return "restricted"
        @unknown default:
            return "unknown"
        }
    }
    switch status {
    case .fullAccess:
        return "full_access"
    case .writeOnly:
        return "write_only"
    case .authorized:
        return "full_access"
    case .notDetermined:
        return "not_determined"
    case .denied:
        return "denied"
    case .restricted:
        return "restricted"
    @unknown default:
        return "unknown"
    }
}

func hasReadAccess(_ type: EKEntityType) -> Bool {
    let status = EKEventStore.authorizationStatus(for: type)
    if #available(macOS 14.0, *) {
        return status == .fullAccess
    }
    return status == .authorized
}

func requestAccess(store: EKEventStore, type: EKEntityType) throws -> Bool {
    let semaphore = DispatchSemaphore(value: 0)
    var granted = false
    var requestError: Error?
    if #available(macOS 14.0, *) {
        if type == .event {
            store.requestFullAccessToEvents { value, error in
                granted = value
                requestError = error
                semaphore.signal()
            }
        } else {
            store.requestFullAccessToReminders { value, error in
                granted = value
                requestError = error
                semaphore.signal()
            }
        }
    } else {
        store.requestAccess(to: type) { value, error in
            granted = value
            requestError = error
            semaphore.signal()
        }
    }
    semaphore.wait()
    if let requestError {
        throw ApplePimError(code: "PERMISSION_REQUEST_FAILED", message: requestError.localizedDescription)
    }
    return granted
}

func requireReadAccess(_ type: EKEntityType, name: String) throws {
    guard hasReadAccess(type) else {
        throw ApplePimError(
            code: "PERMISSION_REQUIRED",
            message: "macOS has not granted full access to \(name). Use the explicit authorization action first."
        )
    }
}

func eventPayload(_ event: EKEvent) -> JsonObject {
    [
        "id": event.eventIdentifier ?? "",
        "title": boundedString(event.title, maximumLength: 1_024),
        "start_at": isoString(event.startDate),
        "end_at": isoString(event.endDate),
        "all_day": event.isAllDay,
        "calendar_id": event.calendar.calendarIdentifier,
        "calendar_title": event.calendar.title,
        "location": boundedString(event.location, maximumLength: 1_024),
        "notes": boundedString(event.notes, maximumLength: 4_096),
        "url": boundedString(event.url?.absoluteString, maximumLength: 2_048),
    ]
}

func reminderPayload(_ reminder: EKReminder) -> JsonObject {
    let dueDate = reminder.dueDateComponents.flatMap { Calendar.current.date(from: $0) }
    let completedDate = reminder.completionDate
    return [
        "id": reminder.calendarItemIdentifier,
        "title": boundedString(reminder.title, maximumLength: 1_024),
        "due_at": isoString(dueDate),
        "completed": reminder.isCompleted,
        "completed_at": isoString(completedDate),
        "calendar_id": reminder.calendar.calendarIdentifier,
        "calendar_title": reminder.calendar.title,
        "notes": boundedString(reminder.notes, maximumLength: 4_096),
        "priority": reminder.priority,
    ]
}

func dateRange(_ input: JsonObject) throws -> (Date, Date) {
    let now = Date()
    let start = try parseDate(stringValue(input, "start_at"), field: "start_at") ?? now.addingTimeInterval(-24 * 60 * 60)
    let end = try parseDate(stringValue(input, "end_at"), field: "end_at") ?? now.addingTimeInterval(30 * 24 * 60 * 60)
    guard end >= start else {
        throw ApplePimError(code: "INVALID_INPUT", message: "end_at must be later than start_at.")
    }
    guard end.timeIntervalSince(start) <= 366 * 24 * 60 * 60 else {
        throw ApplePimError(code: "INVALID_INPUT", message: "The requested range may not exceed 366 days.")
    }
    return (start, end)
}

func reminderCalendar(store: EKEventStore, name: String) throws -> EKCalendar {
    let calendars = store.calendars(for: .reminder)
    if !name.isEmpty {
        let matches = calendars.filter {
            $0.title.compare(name, options: [.caseInsensitive, .diacriticInsensitive]) == .orderedSame
        }
        guard matches.count == 1, let calendar = matches.first else {
            throw ApplePimError(code: "REMINDER_LIST_NOT_FOUND", message: "No unique writable Reminders list matches the requested name.")
        }
        guard calendar.allowsContentModifications else {
            throw ApplePimError(code: "CALENDAR_READ_ONLY", message: "The requested Reminders list is read-only.")
        }
        return calendar
    }
    guard let calendar = store.defaultCalendarForNewReminders() else {
        throw ApplePimError(code: "DEFAULT_REMINDER_LIST_UNAVAILABLE", message: "No default Apple Reminders list is available.")
    }
    guard calendar.allowsContentModifications else {
        throw ApplePimError(code: "CALENDAR_READ_ONLY", message: "The default Reminders list is read-only.")
    }
    return calendar
}

func fetchReminders(store: EKEventStore, predicate: NSPredicate) -> [EKReminder] {
    let semaphore = DispatchSemaphore(value: 0)
    var reminders: [EKReminder] = []
    store.fetchReminders(matching: predicate) { items in
        reminders = items ?? []
        semaphore.signal()
    }
    semaphore.wait()
    return reminders
}

func run(_ input: JsonObject) throws -> JsonObject {
    let action = try stringValue(input, "action", required: true)
    let store = EKEventStore()
    if action == "status" {
        return [
            "ok": true,
            "platform": "macos_eventkit",
            "events_authorization": authorizationName(EKEventStore.authorizationStatus(for: .event)),
            "reminders_authorization": authorizationName(EKEventStore.authorizationStatus(for: .reminder)),
        ]
    }
    if action == "request_access" {
        let requestEvents = boolValue(input, "events", fallback: true)
        let requestReminders = boolValue(input, "reminders", fallback: true)
        guard requestEvents || requestReminders else {
            throw ApplePimError(code: "INVALID_INPUT", message: "Select events, reminders, or both for authorization.")
        }
        var eventsGranted: Bool? = nil
        var remindersGranted: Bool? = nil
        if requestEvents && !hasReadAccess(.event) {
            eventsGranted = try requestAccess(store: store, type: .event)
        }
        if requestReminders && !hasReadAccess(.reminder) {
            remindersGranted = try requestAccess(store: store, type: .reminder)
        }
        return [
            "ok": true,
            "events_granted": eventsGranted as Any,
            "reminders_granted": remindersGranted as Any,
            "events_authorization": authorizationName(EKEventStore.authorizationStatus(for: .event)),
            "reminders_authorization": authorizationName(EKEventStore.authorizationStatus(for: .reminder)),
        ]
    }
    if action == "list_items" {
        let range = try dateRange(input)
        var events: [JsonObject] = []
        var reminders: [JsonObject] = []
        var eventsTruncated = false
        var remindersTruncated = false
        if boolValue(input, "include_events", fallback: true) {
            try requireReadAccess(.event, name: "Calendar")
            let predicate = store.predicateForEvents(withStart: range.0, end: range.1, calendars: nil)
            let matchingEvents = store.events(matching: predicate)
                .sorted { ($0.startDate ?? .distantFuture) < ($1.startDate ?? .distantFuture) }
            eventsTruncated = matchingEvents.count > 200
            events = Array(matchingEvents.prefix(200)).map(eventPayload)
        }
        if boolValue(input, "include_reminders", fallback: true) {
            try requireReadAccess(.reminder, name: "Reminders")
            // A large portion of Apple Reminders have no due date.  Passing a
            // bounded due-date predicate silently excludes them, which makes a
            // connected account appear empty.  Reminder data is therefore an
            // outstanding-task list: return all incomplete items, including
            // undated and overdue items.  The requested date range remains the
            // Calendar-event range.
            let predicate = store.predicateForIncompleteReminders(
                withDueDateStarting: nil,
                ending: nil,
                calendars: nil
            )
            let matchingReminders = fetchReminders(store: store, predicate: predicate)
                .sorted {
                    let left = $0.dueDateComponents.flatMap { Calendar.current.date(from: $0) } ?? .distantFuture
                    let right = $1.dueDateComponents.flatMap { Calendar.current.date(from: $0) } ?? .distantFuture
                    return left < right
                }
            remindersTruncated = matchingReminders.count > 200
            reminders = Array(matchingReminders.prefix(200)).map(reminderPayload)
        }
        return [
            "ok": true,
            "start_at": isoString(range.0),
            "end_at": isoString(range.1),
            "events": events,
            "events_truncated": eventsTruncated,
            "reminders": reminders,
            "reminders_truncated": remindersTruncated,
        ]
    }
    if action == "create_reminder" {
        try requireReadAccess(.reminder, name: "Reminders")
        let title = try stringValue(input, "title", required: true)
        let calendar = try reminderCalendar(
            store: store,
            name: try stringValue(input, "calendar_name")
        )
        let reminder = EKReminder(eventStore: store)
        reminder.title = title
        reminder.calendar = calendar
        reminder.notes = try stringValue(input, "notes")
        reminder.priority = input["priority"] as? Int ?? 0
        if let due = try parseDate(try stringValue(input, "due_at"), field: "due_at") {
            reminder.dueDateComponents = Calendar.current.dateComponents(
                [.calendar, .timeZone, .year, .month, .day, .hour, .minute],
                from: due
            )
        }
        try store.save(reminder, commit: true)
        return ["ok": true, "reminder": reminderPayload(reminder)]
    }
    throw ApplePimError(code: "INVALID_ACTION", message: "Unsupported action: \(action).")
}

do {
    let raw = FileHandle.standardInput.readDataToEndOfFile()
    guard let value = try JSONSerialization.jsonObject(with: raw) as? JsonObject else {
        throw ApplePimError(code: "INVALID_INPUT", message: "Input must be a JSON object.")
    }
    printResponse(try run(value))
} catch let error as ApplePimError {
    printResponse(["ok": false, "error": ["code": error.code, "message": error.message]])
} catch {
    printResponse(["ok": false, "error": ["code": "HOST_SERVICE_FAILED", "message": error.localizedDescription]])
}
