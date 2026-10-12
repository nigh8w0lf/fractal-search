import QtQuick
import Quickshell
import Quickshell.Io
import qs.Commons

Item {
    id: app
    visible: false
    required property string socketPath
    property string serviceName: "fractal-search.service"
    property bool manageService: true
    property string backendExecutable: ""
    property string configuredProvider: "cpu"
    property int configuredDeviceId: -1
    property string cudaLibraryDirectory: ""
    property string runtimeLibraryPath: ""
    readonly property alias runtimeDiagnostics: runtimeDiagnostics
    readonly property alias startAtLogin: autostart.startAtLogin
    readonly property alias loginSettingKnown: autostart.known
    readonly property alias loginSettingBusy: autostart.busy
    property bool starting: false
    property bool installingModels: false
    property bool stoppedCleanly: false
    property int connectAttempts: 0
    property var modelProgress: ({
            stage: "verifying",
            current: 0,
            total: 1,
            filename: ""
        })
    property bool ready: false
    property bool loading: false
    property bool searchStarting: false
    property bool searchWarming: false
    property bool indexing: false
    property bool closing: false
    property bool restarting: false
    property bool previewLoading: false
    property string page: "summon"
    property string previousPage: "summon"
    property int selectedIndex: -1
    property var sourceId: null
    property var mediaKind: null
    property bool resultRequestPending: false
    property int resultRequestSerial: 0
    property int previewGeneration: 0
    property bool previewRequestPending: false
    property bool originalRequestPending: false
    readonly property bool openingFile: originalRequestPending || fileOpener.opening
    property int previewRequestSerial: 0
    property int queryGeneration: 0
    property bool deferredSearch: false
    property var reindexQueue: []
    property bool reindexTextPassages: false
    readonly property var settings: status.settings
    readonly property bool batchRestartRequired: ready && status.activeImageBatchSize !== settings.imageBatchSize
    readonly property color borderColor: Qt.rgba(foregroundColor.r, foregroundColor.g, foregroundColor.b, 0.25)
    readonly property string monoFont: SearchTheme.font.family
    property string errorMessage: ""
    property int indexFailureCount: 0
    property var indexFailures: []
    property bool indexReportAcknowledged: false
    readonly property bool showIndexReport: indexFailureCount > 0 && !indexReportAcknowledged
    readonly property string indexFailureDetails: indexFailures.map(failure => failure.path + "\n" + failure.error).join("\n\n")
    property string notice: ""
    property string query: ""
    property string activeQuery: ""
    property int total: 0
    property var status: ({
            catalog: {
                assetCount: 0,
                fileCount: 0,
                sources: []
            },
            modelsPresent: false,
            provider: "CPU",
            threads: 0,
            activeImageBatchSize: 1,
            modelDirectory: "",
            themeFile: "",
            runtimeDiagnostics: null,
            settings: {
                topMatches: 6,
                resultLayout: "listPreview",
                watchChanges: false,
                closeAfterOpening: true,
                imageBatchSize: 1,
                groupTextByFile: false,
                textIdleSeconds: 900,
                mediaIdleSeconds: 300
            }
        })
    property var progress: ({
            stage: "scanning",
            current: 0,
            total: 0,
            failed: 0,
            filename: ""
        })
    property var selectedAsset: null
    property string previewUrl: ""
    property string previewError: ""
    property var pending: ({})
    property double nextId: 1
    property double discardedBeforeId: 0
    readonly property var colors: ({
            background: Color.popups.background,
            foreground: Color.popups.text,
            accent: Color.accent,
            surface: Qt.alpha(Color.foreground, 0.06),
            muted: Color.muted,
            error: Color.urgent,
            success: Color.accent
        })
    readonly property color foregroundColor: colors.foreground
    readonly property color secondaryText: Qt.rgba(foregroundColor.r, foregroundColor.g, foregroundColor.b, 0.65)
    readonly property bool busy: loading || indexing || installingModels || closing || restarting
    readonly property bool serviceRunning: worker.connected
    readonly property bool serviceStarting: !closing && ((starting && !ready) || searchWarming)
    readonly property string loadingMessage: serviceStarting || searchStarting ? "Starting search service…" : activeQuery.length ? "Searching your library…" : "Loading results…"
    property alias results: results
    signal dismissRequested
    signal stopped
    signal serviceFailed
    signal externalAppOpened
    signal restartRequired

    DesktopFileOpener {
        id: fileOpener
        onFailed: message => app.fail(message)
        onOpened: {
            if (app.settings.closeAfterOpening)
                app.dismiss();
            else
                app.externalAppOpened();
        }
    }
    RuntimeDiagnostics {
        id: runtimeDiagnostics
        executable: app.backendExecutable
        provider: app.configuredProvider
        deviceId: app.configuredDeviceId
        libraryDirectory: app.runtimeLibraryPath
        onFailed: message => console.error("[Fractal Search diagnostics] " + message)
    }
    ListModel {
        id: results
    }
    ServiceAutostart {
        id: autostart
        serviceName: app.serviceName
        enabledForSession: app.manageService
        onFailed: message => app.fail(message)
    }

    function setStartAtLogin(enabled) {
        autostart.setStartAtLogin(enabled);
    }

    function fail(message) {
        errorMessage = String(message);
        console.error("[Fractal Search] " + errorMessage);
    }
    function acknowledgeStatus() {
        if (busy)
            return;
        errorMessage = "";
        notice = "";
        // Acknowledgement hides the banner, not the diagnostic report. New
        // scans reannounce failures; settings keeps the last report available.
        indexReportAcknowledged = true;
    }
    function request(command, callback) {
        if (!worker.connected || (!ready && command.type !== "shutdown")) {
            fail("The search service is not connected.");
            return;
        }
        const id = nextId++;
        const next = Object.assign({}, pending);
        next[id] = {
            type: command.type,
            generation: ["search", "browse", "preview"].indexOf(command.type) >= 0 ? queryGeneration : null,
            previewGeneration: command.type === "preview" ? previewGeneration : null,
            scope: ["search", "browse"].indexOf(command.type) >= 0 ? resultScope() : null,
            callback: callback
        };
        pending = next;
        worker.write(JSON.stringify({
            id: id,
            command: command
        }) + "\n");
        worker.flush();
    }
    function receive(line) {
        try {
            const message = JSON.parse(line);
            // IDs are monotonic across connections; ignore buffered replies from a discarded socket.
            if (typeof message.id === "number" && message.id < discardedBeforeId)
                return;
            if (message.event === "ready") {
                if (message.payload.protocolVersion !== 1)
                    throw new Error("Incompatible backend protocol. Reinstall Fractal Search.");
                status = message.payload;
                searchWarming = status.textModelWarming === true;
                if (status.warmupError)
                    fail(status.warmupError);
                ready = true;
                starting = false;
                connectTimer.stop();
                const job = status.activeJob;
                indexing = !!job && job.kind === "index";
                installingModels = !!job && job.kind === "installModels";
                if (indexing) {
                    if (job.progress)
                        progress = job.progress;
                    indexFailures = job.failures;
                    indexFailureCount = job.progress ? job.progress.failed : 0;
                    notice = "Indexing continues in the background…";
                }
                if (installingModels && job.progress)
                    modelProgress = job.progress;
                if (status.lastIndexReport)
                    restoreReport(status.lastIndexReport);
                if (closing) {
                    request({
                        type: "shutdown"
                    }, function () {});
                    return;
                }
                if (!indexing && !installingModels) {
                    deferredSearch = false;
                    browse(false, false);
                }
                return;
            }
            if (message.event === "status") {
                status = message.payload;
                return;
            }
            if (message.event === "fatal") {
                searchWarming = false;
                starting = false;
                connectTimer.stop();
                fail(message.payload.error);
                serviceFailed();
                return;
            }
            if (message.event === "stopped") {
                searchWarming = false;
                stoppedCleanly = message.payload.clean;
                if (!restarting)
                    closing = false;
                ready = false;
                starting = false;
                connectTimer.stop();
                if (stoppedCleanly)
                    worker.connected = false;
                if (!stoppedCleanly) {
                    fail("The search service did not stop cleanly. Check journalctl --user -u " + serviceName);
                    serviceFailed();
                } else if (!restarting) {
                    notice = "Search service stopped.";
                    dismiss();
                    stopped();
                }
                return;
            }
            if (message.event === "modelProgress") {
                installingModels = true;
                modelProgress = message.payload;
                return;
            }
            if (message.event === "warmupFinished") {
                searchWarming = false;
                if (!message.payload.ok)
                    fail(message.payload.error);
                return;
            }
            if (message.event === "modelsFinished") {
                finishModelSetup(message.payload.ok ? message.payload.result : null, message.payload.error || "");
                return;
            }
            if (message.event === "indexStarted") {
                indexing = true;
                indexReportAcknowledged = false;
                errorMessage = "";
                indexFailureCount = 0;
                indexFailures = [];
                notice = "Folder changed · updating index…";
                return;
            }
            if (message.event === "indexFinished") {
                indexing = false;
                indexReportAcknowledged = false;
                indexFailureCount = message.payload.failed;
                indexFailures = message.payload.failures;
                notice = "Index updated · " + message.payload.indexed + " file(s) updated";
                if (indexFailureCount)
                    fail(indexFailureCount + " file(s) skipped. Open Skipped files for details.");
                browse(false, false);
                return;
            }
            if (message.event === "indexError") {
                indexing = false;
                fail(message.payload.directory + ": " + message.payload.error);
                browse(false, false);
                return;
            }
            if (message.event === "indexProgress") {
                indexing = true;
                progress = message.payload;
                indexReportAcknowledged = false;
                indexFailureCount = progress.failed;
                if (progress.failure) {
                    // Match the service's bounded report, including when a
                    // later model/storage error prevents a final result.
                    if (indexFailures.length < 100)
                        indexFailures = indexFailures.concat([progress.failure]);
                    fail(indexFailureCount + " file(s) skipped. Latest: " + progress.failure.path + "\n" + progress.failure.error);
                }
                return;
            }
            if (message.event === "searchProgress") {
                const entry = pending[message.id];
                if (!entry || entry.type !== "search")
                    throw new Error("Unexpected search progress " + message.id);
                if (!isCurrentReply(entry))
                    return;
                if (message.payload.stage === "starting")
                    searchStarting = true;
                else if (message.payload.stage === "searching")
                    searchStarting = false;
                else
                    throw new Error("Unknown search stage " + message.payload.stage);
                return;
            }
            if (message.id === null && !message.ok) {
                fail(message.error);
                return;
            }
            const entry = pending[message.id];
            if (!entry)
                throw new Error("Unexpected service reply " + message.id);
            const next = Object.assign({}, pending);
            delete next[message.id];
            pending = next;
            if (!message.ok && isCurrentReply(entry))
                fail(message.error);
            if (entry.callback)
                entry.callback(message.ok ? message.result : null, message.ok ? "" : message.error);
        } catch (error) {
            loading = false;
            searchStarting = false;
            fail("Cannot read search service response: " + error);
        }
    }
    function refreshStatus() {
        // Performance polls only while visible. Coalesce refresh buttons and
        // result-triggered refreshes too, so a slow service cannot grow a queue.
        if (!ready || !worker.connected || closing || Object.keys(pending).some(id => pending[id].type === "status"))
            return;
        request({
            type: "status"
        }, function (result) {
            if (result)
                app.status = result;
        });
    }
    function restoreReport(report) {
        indexFailures = report.failures;
        indexFailureCount = report.failed;
        indexReportAcknowledged = false;
        if (report.error)
            fail(report.error);
        else if (report.failed)
            fail(report.failed + " file(s) skipped. Open Skipped files for details.");
    }
    function installModels() {
        if (!ready || busy)
            return;
        installingModels = true;
        errorMessage = "";
        notice = "";
        modelProgress = {
            stage: "verifying",
            current: 0,
            total: 1,
            filename: ""
        };
        request({
            type: "installModels"
        }, finishModelSetup);
    }
    function finishModelSetup(result, error) {
        installingModels = false;
        if (result) {
            status = result;
            notice = "Search model verified and ready.";
            browse(false, false);
        } else
            fail(error);
    }
    function ensureService() {
        if (worker.connected || starting || closing)
            return;
        starting = true;
        stoppedCleanly = false;
        connectAttempts = 0;
        errorMessage = "";
        notice = "";
        resetConnectionRequests();
        if (manageService)
            starter.running = true;
        else
            connectTimer.start();
    }
    function resultScope() {
        return JSON.stringify([activeQuery, sourceId, mediaKind]);
    }
    function isCurrentReply(entry) {
        return (entry.generation === null || entry.generation === queryGeneration)
            && (entry.scope === null || entry.scope === resultScope())
            && (entry.previewGeneration === null || entry.previewGeneration === previewGeneration);
    }
    function clearResults() {
        ++previewGeneration;
        previewTimer.stop();
        previewLoading = false;
        selectedIndex = -1;
        selectedAsset = null;
        previewUrl = "";
        previewError = "";
        total = 0;
        results.clear();
    }
    function resetConnectionRequests() {
        originalRequestPending = false;
        // Old socket callbacks cannot clear or overwrite a newer connection's work.
        ++queryGeneration;
        ++resultRequestSerial;
        ++previewGeneration;
        ++previewRequestSerial;
        previewRequestPending = false;
        resultRequestPending = false;
        previewLoading = false;
        discardedBeforeId = nextId;
        pending = {};
    }
    function editQuery(text) {
        query = text;
        activeQuery = text.trim();
        deferredSearch = false;
        ++queryGeneration;
        loading = false;
        searchStarting = false;
        clearResults();
    }
    function setMediaKind(kind) {
        mediaKind = kind;
        browse(false);
    }
    function setSourceId(id) {
        sourceId = id;
        browse(false);
    }
    function browse(append, clearError) {
        if (closing)
            return;
        // Generation and the complete query/folder/type scope are both checked.
        // Invalidate immediately, even when a request must wait for warmup or IO.
        const generation = append ? queryGeneration : ++queryGeneration;
        if (!append)
            clearResults();
        loading = true;
        searchStarting = false;
        if (clearError !== false)
            errorMessage = "";
        if (indexing || installingModels || !ready || resultRequestPending) {
            deferredSearch = true;
            return;
        }
        deferredSearch = false;
        const scope = resultScope();
        const serial = ++resultRequestSerial;
        const offset = append ? results.count : 0;
        const command = activeQuery.length ? {
            type: "search",
            query: activeQuery,
            sourceId: sourceId,
            mediaKind: mediaKind,
            maxResults: settings.topMatches,
            offset: offset,
            limit: settings.topMatches
        } : {
            type: "browse",
            sourceId: sourceId,
            mediaKind: mediaKind,
            offset: offset,
            limit: settings.topMatches
        };
        resultRequestPending = true;
        request(command, function (result) {
            if (serial !== app.resultRequestSerial)
                return;
            app.resultRequestPending = false;
            if (app.deferredSearch) {
                app.deferredSearch = false;
                app.browse(false, false);
                return;
            }
            if (generation !== app.queryGeneration || scope !== app.resultScope())
                return;
            app.loading = false;
            app.searchStarting = false;
            if (!result)
                return;
            app.total = result.total;
            for (const asset of result.items) {
                const row = app.resultAsset(asset, true);
                if (!row)
                    return;
                results.append(row);
            }
            if (!append && results.count)
                app.select(0);
            app.refreshStatus();
        });
    }
    function boundedUtf8(value, maximum) {
        if (typeof value !== "string")
            return false;
        let bytes = 0;
        for (let i = 0; i < value.length; i++) {
            const code = value.charCodeAt(i);
            if (code < 0x80) bytes += 1;
            else if (code < 0x800) bytes += 2;
            else if (code >= 0xD800 && code <= 0xDBFF) {
                const low = value.charCodeAt(++i);
                if (!(low >= 0xDC00 && low <= 0xDFFF)) return false;
                bytes += 4;
            } else if (code >= 0xDC00 && code <= 0xDFFF) return false;
            else bytes += 3;
            if (bytes > maximum) return false;
        }
        return true;
    }
    function resultAsset(asset, allowAdditional) {
        if (!asset || !asset.assetId || ["image", "text", "audio", "video"].indexOf(asset.mediaKind) < 0 || (isVisualAsset(asset) && !asset.thumbnailUrl)) {
            fail("Service returned incomplete media metadata. Reinstall matching UI and backend versions.");
            return null;
        }
        const hasScore = asset.similarity !== null && asset.similarity !== undefined;
        const hasHybridScore = asset.hybridScore !== null && asset.hybridScore !== undefined;
        if ((hasScore && !Number.isFinite(asset.similarity)) || (hasHybridScore && (!Number.isFinite(asset.hybridScore) || mediaKind !== "text" || !activeQuery.length))) {
            fail("Service returned an invalid search score. Reinstall matching UI and backend versions.");
            return null;
        }
        if (asset.textExcerpt != null && !boundedUtf8(asset.textExcerpt, 2048)) {
            fail("Service returned an invalid text excerpt. Reinstall matching UI and backend versions.");
            return null;
        }
        const additional = asset.additionalMatches == null ? [] : asset.additionalMatches;
        if (!Array.isArray(additional) || additional.length > 5 || (additional.length && (!allowAdditional || asset.mediaKind !== "text" || mediaKind !== "text" || !activeQuery.length))) {
            fail("Service returned invalid grouped text matches. Reinstall matching UI and backend versions.");
            return null;
        }
        const matches = [];
        for (const match of additional) {
            if (!match || match.mediaKind !== "text" || match.path !== asset.path || match.assetId === asset.assetId || matches.some(item => item.assetId === match.assetId)) {
                fail("Service returned inconsistent grouped text matches. Reinstall matching UI and backend versions.");
                return null;
            }
            const row = resultAsset(match, false);
            if (!row)
                return null;
            matches.push(row);
        }
        return {
            assetId: asset.assetId,
            filename: asset.filename,
            path: asset.path,
            mediaKind: asset.mediaKind,
            segmentIndex: asset.segmentIndex,
            startMs: asset.startMs == null ? -1 : asset.startMs,
            endMs: asset.endMs == null ? -1 : asset.endMs,
            startByte: asset.startByte == null ? -1 : asset.startByte,
            endByte: asset.endByte == null ? -1 : asset.endByte,
            startLine: asset.startLine == null ? -1 : asset.startLine,
            endLine: asset.endLine == null ? -1 : asset.endLine,
            textExcerpt: asset.textExcerpt || "",
            thumbnailUrl: asset.thumbnailUrl || "",
            width: asset.width,
            height: asset.height,
            similarity: hasScore ? asset.similarity : 0,
            hasScore: hasScore,
            hybridScore: hasHybridScore ? asset.hybridScore : 0,
            hasHybridScore: hasHybridScore,
            // Keep this bounded collection a JSON value rather than a mutable
            // nested QML ListModel; it survives selection and result replacement.
            additionalMatchesJson: JSON.stringify(matches),
            additionalMatchCount: matches.length
        };
    }
    function additionalMatches(asset) {
        return asset && asset.additionalMatchesJson ? JSON.parse(asset.additionalMatchesJson) : [];
    }
    function search(text) {
        if (closing)
            return;
        page = "results";
        query = text;
        activeQuery = text.trim();
        browse(false);
    }
    function scoreLabel(asset) {
        if (!asset)
            return "";
        if (asset.hasHybridScore)
            return "Hybrid score " + asset.hybridScore.toFixed(5);
        return asset.hasScore ? "Cosine " + asset.similarity.toFixed(3) : "";
    }
    function indexFolder(directory, rebuildText) {
        if (busy || !ready)
            return;
        indexing = true;
        indexReportAcknowledged = false;
        errorMessage = "";
        indexFailureCount = 0;
        indexFailures = [];
        notice = "";
        progress = {
            stage: "scanning",
            current: 0,
            total: 0,
            failed: 0,
            filename: ""
        };
        const command = {type: "index", directory: String(directory)};
        if (rebuildText === true) {
            command.reindexText = true;
            command.verifyContent = true;
        }
        request(command, function (result) {
            app.indexing = false;
            if (app.closing)
                return;
            if (result) {
                app.indexFailureCount = result.failed;
                app.indexFailures = result.failures;
                app.notice = (result.cancelled ? "Indexing stopped. " : result.failed ? "Indexing finished with errors. " : "Folder indexed. ") + result.indexed + " updated · " + result.unchanged + " unchanged" + (result.failed ? " · " + result.failed + " skipped" : "") + (result.removed ? " · " + result.removed + " removed from index" : "");
                if (result.failed)
                    app.errorMessage = result.failed + " file(s) could not be indexed. Open Skipped files for the filenames and errors, then rescan after repairing them.";
                app.activeQuery = "";
                app.query = "";
                if (app.reindexQueue.length && !result.cancelled && !result.failed)
                    app.indexNext();
                else {
                    if (app.reindexQueue.length && result.failed)
                        app.notice += " · Reindex paused; repair skipped files before continuing.";
                    app.reindexQueue = [];
                    app.browse(false, false);
                }
            } else {
                app.reindexQueue = [];
                // Completed files remain usable after a scan fails. Refresh
                // the gallery while preserving the visible indexing error.
                app.activeQuery = "";
                app.query = "";
                app.browse(false, false);
            }
        });
    }
    function cancelIndex() {
        reindexQueue = [];
        request({
            type: "cancel"
        }, function (result) {
            if (result)
                app.notice = "Finishing active work and saving completed files…";
        });
    }
    function setModelDirectory(directory) {
        loading = true;
        errorMessage = "";
        notice = "Verifying model files…";
        request({
            type: "setModelDirectory",
            directory: String(directory)
        }, function (result) {
            app.loading = false;
            app.notice = "";
            if (result) {
                app.status = result;
                app.notice = "Model files verified. Ready to index folders.";
            }
        });
    }
    function shellQuote(value) {
        return "'" + value.replace(/'/g, "'\\''") + "'";
    }
    function folderPath(path) {
        return path.substring(0, path.lastIndexOf("/"));
    }
    function displayPath(path) {
        const home = Quickshell.env("HOME");
        return home && (path === home || path.startsWith(home + "/")) ? "~" + path.substring(home.length) : path;
    }
    function isVisualAsset(asset) {
        return !!asset && (asset.mediaKind === "image" || asset.mediaKind === "video");
    }
    function timeLabel(milliseconds) {
        const seconds = Math.max(0, Math.floor(milliseconds / 1000));
        const minutes = Math.floor(seconds / 60);
        return minutes + ":" + (seconds % 60 < 10 ? "0" : "") + (seconds % 60);
    }
    function mediaLabel(asset) {
        if (!asset)
            return "";
        const kind = asset.mediaKind.charAt(0).toUpperCase() + asset.mediaKind.slice(1);
        if (asset.mediaKind === "text") {
            const passage = kind + " · passage " + (asset.segmentIndex + 1);
            if (asset.startLine > 0 && asset.endLine >= asset.startLine)
                return passage + " · " + (asset.startLine === asset.endLine ? "line " + asset.startLine : "lines " + asset.startLine + "–" + asset.endLine);
            if (asset.startByte >= 0 && asset.endByte >= asset.startByte)
                return passage + " · bytes " + asset.startByte + "–" + asset.endByte;
            return passage;
        }
        if (asset.mediaKind === "audio" || asset.mediaKind === "video")
            return kind + " · " + timeLabel(asset.startMs) + "–" + timeLabel(asset.endMs);
        return kind;
    }
    function previewText() {
        if (!selectedAsset)
            return "Select a result";
        if (selectedAsset.mediaKind === "text")
            return selectedAsset.textExcerpt;
        if (selectedAsset.mediaKind === "audio")
            return mediaLabel(selectedAsset) + "\n\nOpen file to listen in your default app. Playback starts according to that app; segment seeking is not requested.";
        return "";
    }
    function select(index) {
        if (index < 0 || index >= results.count)
            return;
        selectedIndex = index;
        const asset = results.get(index);
        selectedAsset = {
            assetId: asset.assetId,
            filename: asset.filename,
            path: asset.path,
            mediaKind: asset.mediaKind,
            segmentIndex: asset.segmentIndex,
            startMs: asset.startMs,
            endMs: asset.endMs,
            startByte: asset.startByte,
            endByte: asset.endByte,
            startLine: asset.startLine,
            endLine: asset.endLine,
            additionalMatchesJson: asset.additionalMatchesJson,
            additionalMatchCount: asset.additionalMatchCount,
            textExcerpt: asset.textExcerpt,
            thumbnailUrl: asset.thumbnailUrl,
            width: asset.width,
            height: asset.height,
            similarity: asset.similarity,
            hasScore: asset.hasScore,
            hybridScore: asset.hybridScore,
            hasHybridScore: asset.hasHybridScore
        };
        ++previewGeneration;
        previewLoading = asset.mediaKind === "image" && previewRequestPending;
        previewTimer.stop();
        previewUrl = asset.mediaKind === "video" ? asset.thumbnailUrl : "";
        previewError = "";
        if (asset.mediaKind === "image")
            previewTimer.restart();
    }
    function preview(asset) {
        for (let i = 0; i < results.count; ++i) {
            if (results.get(i).assetId === asset.assetId) {
                select(i);
                return;
            }
        }
        fail("Cannot preview a file outside the current results.");
    }
    function loadPreview() {
        if (!selectedAsset || selectedAsset.mediaKind !== "image" || previewRequestPending || closing)
            return;
        const assetId = selectedAsset.assetId;
        const generation = queryGeneration;
        const selection = previewGeneration;
        const serial = ++previewRequestSerial;
        previewRequestPending = true;
        previewLoading = true;
        request({
            type: "preview",
            assetId: assetId
        }, function (result, error) {
            if (serial !== app.previewRequestSerial)
                return;
            app.previewRequestPending = false;
            if (generation !== app.queryGeneration || selection !== app.previewGeneration || !app.selectedAsset || app.selectedAsset.assetId !== assetId) {
                // One preview at a time; dispatch only the latest selected image next.
                previewTimer.restart();
                return;
            }
            app.previewLoading = false;
            if (result)
                app.previewUrl = result.url;
            else
                app.previewError = error;
        });
    }
    function openOriginal() {
        if (!selectedAsset || !ready || closing)
            return;
        if (openingFile) {
            fail("A file is already being opened. Wait before trying again.");
            return;
        }
        originalRequestPending = true;
        request({
            type: "originalUrl",
            assetId: selectedAsset.assetId
        }, function (result) {
            app.originalRequestPending = false;
            if (!result)
                return;
            fileOpener.open(result.url);
        });
    }
    function revealOriginal() {
        if (!selectedAsset || !ready || closing)
            return;
        request({
            type: "reveal",
            assetId: selectedAsset.assetId
        }, function (result) {
            if (result)
                app.externalAppOpened();
        });
    }
    function dismiss() {
        page = "summon";
        editQuery("");
        dismissRequested();
    }

    function showSettings() {
        autostart.refresh();
        previousPage = page;
        page = "settings";
    }
    function back() {
        if (page === "settings")
            page = previousPage;
        else if (page === "results") {
            page = "summon";
            editQuery("");
        } else
            dismiss();
    }
    function updateSettings(changes) {
        if (!ready || busy)
            return;
        loading = true;
        request({
            type: "updateSettings",
            settings: Object.assign({}, settings, changes)
        }, function (result) {
            app.loading = false;
            if (result) {
                app.status = result;
                if (changes.imageBatchSize !== undefined && app.batchRestartRequired)
                    app.restartRequired();
                if (changes.topMatches !== undefined || changes.groupTextByFile !== undefined)
                    app.browse(false);
            }
        });
    }
    function removeSource(id) {
        if (!ready || busy)
            return;
        loading = true;
        request({
            type: "removeSource",
            sourceId: id
        }, function (result) {
            app.loading = false;
            if (result) {
                app.sourceId = null;
                app.refreshStatus();
                app.browse(false);
            }
        });
    }
    function reindexAll(rebuildText) {
        if (busy || !ready || !status.catalog.sources.length)
            return;
        reindexTextPassages = rebuildText === true;
        reindexQueue = status.catalog.sources.map(source => source.directory);
        indexNext();
    }
    function indexNext() {
        if (!reindexQueue.length || closing)
            return;
        const directory = reindexQueue[0];
        reindexQueue = reindexQueue.slice(1);
        indexFolder(directory, reindexTextPassages);
    }
    Timer {
        id: previewTimer
        interval: 100
        onTriggered: app.loadPreview()
    }
    function shutdown() {
        if (closing)
            return;
        if (!worker.connected) {
            if (starting) {
                closing = true;
                notice = "Stopping the starting search service…";
                if (manageService) {
                    starting = false;
                    connectTimer.stop();
                    stopper.running = true;
                }
                return;
            }
            dismiss();
            stopped();
            return;
        }
        closing = true;
        notice = "Finishing active work and stopping the search service…";
        request({
            type: "shutdown"
        }, function () {});
    }
    function restartService() {
        if (closing || restarting)
            return;
        if (!manageService) {
            fail("Restart the isolated test backend to apply Image batch size.");
            return;
        }
        restarting = true;
        closing = true;
        notice = "Finishing active work and restarting the search service…";
        restarter.running = true;
    }
    Timer {
        id: connectTimer
        interval: 250
        repeat: true
        onTriggered: {
            if (++app.connectAttempts > 80) {
                stop();
                app.starting = false;
                app.fail("Cannot connect to the search service. Check journalctl --user -u " + app.serviceName + " and Retry.");
                app.serviceFailed();
                return;
            }
            if (!worker.connected)
                worker.connected = true;
        }
    }
    Process {
        id: restarter
        command: ["systemctl", "--user", "restart", app.serviceName]
        stderr: StdioCollector {
            id: restartError
            waitForEnd: true
        }
        // qmllint disable signal-handler-parameters
        onExited: exitCode => {
            app.restarting = false;
            app.closing = false;
            if (exitCode !== 0) {
                app.notice = "";
                app.fail("Restarting the search service failed (code " + exitCode + "): " + restartError.text.trim());
                app.serviceFailed();
                return;
            }
            app.ready = false;
            worker.connected = false;
            app.loading = false;
            app.indexing = false;
            app.installingModels = false;
            app.resetConnectionRequests();
            app.notice = "";
            app.starting = true;
            app.stoppedCleanly = false;
            app.connectAttempts = 0;
            connectTimer.start();
        }
        // qmllint enable signal-handler-parameters
    }
    Process {
        id: starter
        command: ["systemctl", "--user", "start", app.serviceName]
        stderr: SplitParser {
            onRead: data => console.error("[Fractal Search service] " + data)
        }
        // qmllint disable signal-handler-parameters
        onExited: exitCode => {
            if (exitCode !== 0) {
                app.starting = false;
                app.fail("Starting " + app.serviceName + " failed (code " + exitCode + "). Check the user service journal and reinstall if missing.");
                app.serviceFailed();
            } else if (!app.closing && !app.stoppedCleanly)
                connectTimer.start();
        }
        // qmllint enable signal-handler-parameters
    }
    Process {
        id: stopper
        command: ["systemctl", "--user", "stop", app.serviceName]
        stderr: SplitParser {
            onRead: data => console.error("[Fractal Search stop] " + data)
        }
        // qmllint disable signal-handler-parameters
        onExited: exitCode => {
            app.closing = false;
            if (exitCode === 0) {
                app.stoppedCleanly = true;
                app.ready = false;
                worker.connected = false;
                app.dismiss();
                app.stopped();
            } else {
                app.fail("Stopping the search service failed (code " + exitCode + "). Check the user service journal.");
                app.serviceFailed();
            }
        }
        // qmllint enable signal-handler-parameters
    }
    Socket {
        id: worker
        path: app.socketPath
        parser: SplitParser {
            onRead: data => app.receive(data)
        }
        onConnectedChanged: {
            if (!connected && app.ready) {
                app.resetConnectionRequests();
                app.ready = false;
                app.loading = false;
                app.searchStarting = false;
                app.indexing = false;
                app.searchWarming = false;
                app.installingModels = false;
                if (!app.stoppedCleanly && !app.restarting) {
                    app.closing = false;
                    app.fail("Search service disconnected. Check journalctl --user -u " + app.serviceName + " and Retry.");
                    app.serviceFailed();
                }
            }
        }
        // Connection refusal while systemd creates the socket is bounded by
        // connectTimer. Once connected, errors are reported immediately.
        // qmllint disable signal-handler-parameters
        onError: error => {
            if (app.restarting)
                return;
            if (app.stoppedCleanly)
                return;
            if (app.starting) {
                console.log("[Fractal Search] Waiting for the service socket (" + error + ").");
                return;
            }
            if (!app.starting) {
                app.fail("Search socket error " + error + ". Check the service journal and Retry.");
                app.serviceFailed();
            }
        }
        // qmllint enable signal-handler-parameters
    }
}
