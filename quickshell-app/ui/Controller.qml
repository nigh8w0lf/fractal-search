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
    property int queryGeneration: 0
    property bool deferredSearch: false
    property string pendingSearchText: ""
    property var reindexQueue: []
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
                imageCount: 0,
                sources: []
            },
            modelsPresent: false,
            provider: "CPU",
            threads: 0,
            activeImageBatchSize: 1,
            modelDirectory: "",
            themeFile: "",
            settings: {
                topMatches: 6,
                resultLayout: "listPreview",
                watchChanges: false,
                closeAfterOpening: true,
                imageBatchSize: 1
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
    readonly property string loadingMessage: serviceStarting || searchStarting ? "Starting search service…" : activeQuery.length ? "Searching your images…" : "Loading images…"
    property alias results: results
    signal dismissRequested
    signal stopped
    signal serviceFailed
    signal externalAppOpened
    signal restartRequired

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
                if (!indexing && !installingModels)
                    browse(false, false);
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
                notice = "Index updated · " + message.payload.indexed + " image(s) updated";
                if (indexFailureCount)
                    fail(indexFailureCount + " image(s) skipped. Open Skipped files for details.");
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
                    fail(indexFailureCount + " image(s) skipped. Latest: " + progress.failure.path + "\n" + progress.failure.error);
                }
                return;
            }
            if (message.event === "searchProgress") {
                const entry = pending[message.id];
                if (!entry || entry.type !== "search")
                    throw new Error("Unexpected search progress " + message.id);
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
            if (!message.ok)
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
            fail(report.failed + " image(s) skipped. Open Skipped files for details.");
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
        pending = {};
        if (manageService)
            starter.running = true;
        else
            connectTimer.start();
    }
    function browse(append, clearError) {
        if (indexing || installingModels || closing || !ready)
            return;
        loading = true;
        searchStarting = false;
        if (clearError !== false)
            errorMessage = "";
        const offset = append ? results.count : 0;
        const generation = ++queryGeneration;
        if (!append) {
            selectedIndex = -1;
            selectedAsset = null;
            previewUrl = "";
            results.clear();
        }
        const command = activeQuery.length ? {
            type: "search",
            query: activeQuery,
            sourceId: sourceId,
            maxResults: settings.topMatches,
            offset: offset,
            limit: settings.topMatches
        } : {
            type: "browse",
            sourceId: sourceId,
            offset: offset,
            limit: settings.topMatches
        };
        request(command, function (result) {
            if (generation !== app.queryGeneration)
                return;
            app.loading = false;
            app.searchStarting = false;
            if (app.deferredSearch) {
                app.deferredSearch = false;
                app.search(app.pendingSearchText);
                return;
            }
            if (!result)
                return;
            app.total = result.total;
            for (const image of result.items) {
                if (!image.assetId || !image.thumbnailUrl) {
                    app.fail("Service returned an image without its thumbnail.");
                    return;
                }
                results.append({
                    assetId: image.assetId,
                    filename: image.filename,
                    path: image.path,
                    thumbnailUrl: image.thumbnailUrl,
                    width: image.width,
                    height: image.height,
                    similarity: image.similarity === null ? 0 : image.similarity,
                    hasScore: image.similarity !== null
                });
            }
            if (!append && results.count)
                app.select(0);
            app.refreshStatus();
        });
    }
    function search(text) {
        if (!ready && serviceStarting) {
            query = text;
            pendingSearchText = text;
            deferredSearch = true;
            return;
        }
        if (indexing || installingModels || closing || !ready)
            return;
        page = "results";
        query = text;
        if (loading) {
            pendingSearchText = text;
            deferredSearch = true;
            return;
        }
        activeQuery = text.trim();
        browse(false);
    }
    function indexFolder(directory) {
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
        request({
            type: "index",
            directory: String(directory)
        }, function (result) {
            app.indexing = false;
            if (app.closing)
                return;
            if (result) {
                app.indexFailureCount = result.failed;
                app.indexFailures = result.failures;
                app.notice = (result.cancelled ? "Indexing stopped. " : result.failed ? "Indexing finished with errors. " : "Folder indexed. ") + result.indexed + " updated · " + result.unchanged + " unchanged" + (result.failed ? " · " + result.failed + " skipped" : "") + (result.removed ? " · " + result.removed + " removed from index" : "");
                if (result.failed)
                    app.errorMessage = result.failed + " image(s) could not be indexed. Open Skipped files for the filenames and errors, then rescan after repairing them.";
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
                // Completed images remain usable after a scan fails. Refresh
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
                app.notice = "Stopping after the active image is saved…";
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
    function select(index) {
        if (index < 0 || index >= results.count)
            return;
        selectedIndex = index;
        const asset = results.get(index);
        selectedAsset = {
            assetId: asset.assetId,
            filename: asset.filename,
            path: asset.path,
            width: asset.width,
            height: asset.height
        };
        previewUrl = "";
        previewError = "";
        previewTimer.restart();
    }
    function preview(asset) {
        for (let i = 0; i < results.count; ++i) {
            if (results.get(i).assetId === asset.assetId) {
                select(i);
                return;
            }
        }
        fail("Cannot preview an image outside the current results.");
    }
    function loadPreview() {
        if (!selectedAsset || previewLoading || closing)
            return;
        const assetId = selectedAsset.assetId;
        previewLoading = true;
        request({
            type: "preview",
            assetId: assetId
        }, function (result, error) {
            app.previewLoading = false;
            if (!app.selectedAsset || app.selectedAsset.assetId !== assetId) {
                previewTimer.restart();
                return;
            }
            if (result)
                app.previewUrl = result.url;
            else
                app.previewError = error;
        });
    }
    function openOriginal() {
        if (!selectedAsset || !ready || closing)
            return;
        request({
            type: "originalUrl",
            assetId: selectedAsset.assetId
        }, function (result) {
            if (!result)
                return;
            if (!Qt.openUrlExternally(result.url))
                app.fail("The desktop could not open this image.");
            else {
                // Hiding releases focus itself. Lower only a panel that stays
                // visible, so hide/resummon cannot race a native layer change.
                if (app.settings.closeAfterOpening)
                    app.dismiss();
                else
                    app.externalAppOpened();
            }
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
        query = "";
        deferredSearch = false;
        dismissRequested();
    }

    function showSettings() {
        deferredSearch = false;
        autostart.refresh();
        previousPage = page;
        page = "settings";
    }
    function back() {
        if (page === "settings")
            page = previousPage;
        else if (page === "results") {
            page = "summon";
            query = "";
            deferredSearch = false;
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
                if (changes.topMatches !== undefined)
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
    function reindexAll() {
        if (busy || !ready || !status.catalog.sources.length)
            return;
        reindexQueue = status.catalog.sources.map(source => source.directory);
        indexNext();
    }
    function indexNext() {
        if (!reindexQueue.length || closing)
            return;
        const directory = reindexQueue[0];
        reindexQueue = reindexQueue.slice(1);
        indexFolder(directory);
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
            app.pending = {};
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
