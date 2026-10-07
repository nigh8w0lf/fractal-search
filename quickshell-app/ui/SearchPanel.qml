pragma ComponentBehavior: Bound
import QtQuick
import qs.Commons
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import Quickshell
import Quickshell.Io
import Quickshell.Wayland
import Quickshell.Hyprland

Item {
    id: panel
    property var shell: null
    property var manifest: null
    property string omarchyPath: Quickshell.env("OMARCHY_PATH")
    readonly property bool opened: window.visible
    property bool manageService: true
    property string socketPath: runtimeConfig.socketPath
    property string serviceName: runtimeConfig.serviceName
    property string pickerExecutable: runtimeConfig.pickerExecutable
    readonly property bool closeShortcutEnabled: runtimeConfig.closeShortcutEnabled
    property int testWidth: 0
    property int testHeight: 0
    RuntimeConfig {
        id: runtimeConfig
    }
    Controller {
        id: app
        socketPath: panel.socketPath
        serviceName: panel.serviceName
        manageService: panel.manageService
        backendExecutable: runtimeConfig.backendExecutable
        configuredProvider: runtimeConfig.configuredProvider
        configuredDeviceId: runtimeConfig.configuredDeviceId
        cudaLibraryDirectory: runtimeConfig.cudaLibraryDirectory
        runtimeLibraryPath: runtimeConfig.runtimeLibraryPath
    }
    HyprlandFocusGrab {
        windows: [window]
        active: window.visible && window.surfaceReady && window.exclusiveKeyboard && !imageFolder.visible && !modelFolder.visible && !app.closing
        // Exclusive layer-shell focus captures mouse input outside the mask
        // on Hyprland. A popup grab keeps typing focused but releases on click.
        onCleared: if (window.visible && window.exclusiveKeyboard && !imageFolder.visible && !modelFolder.visible && !app.closing)
            app.dismiss()
    }
    function show() {
        app.ensureService();
        // Restore the overlay before mapping a panel previously lowered for
        // an external viewer; its first frame must not start behind that app.
        window.exclusiveKeyboard = true;
        window.visible = true;
        focusSearch();
    }
    function focusSearch() {
        window.exclusiveKeyboard = true;
        if (app.page !== "settings")
            searchField.forceActiveFocus();
    }
    function submit() {
        queryTimer.stop();
        if (app.page === "results" && app.query.trim() === app.activeQuery && app.selectedAsset && !app.loading)
            app.openOriginal();
        else
            app.search(searchField.text);
    }
    function navigate(delta) {
        if (app.page !== "results" || !app.results.count)
            return;
        app.select(Math.max(0, Math.min(app.results.count - 1, app.selectedIndex + delta)));
        resultsList.positionViewAtIndex(app.selectedIndex, ListView.Contain);
    }
    Connections {
        target: app
        function onServiceFailed(): void {
            console.error("[Fractal Search] " + app.errorMessage);
        }
        function onRestartRequired(): void {
            panel.requestRestart();
        }
        function onDismissRequested(): void {
            queryTimer.stop();
            // Popups belong to this invocation of the launcher; retaining one
            // after hiding would reopen it and steal focus on the next summon.
            if (skippedFiles.visible)
                skippedFiles.close();
            if (errorDialog.visible)
                errorDialog.close();
            if (removeDialog.visible)
                removeDialog.close();
            if (stopDialog.visible)
                stopDialog.close();
            if (restartDialog.visible)
                restartDialog.close();
            if (imageFolder.visible)
                imageFolder.close();
            if (modelFolder.visible)
                modelFolder.close();
            window.visible = false;
        }
        function onExternalAppOpened(): void {
            window.exclusiveKeyboard = false;
        }
        function onPageChanged(): void {
            if (app.page !== "results")
                queryTimer.stop();
            panel.testWidth = 0;
            panel.testHeight = 0;
            if (app.page !== "settings")
                searchField.forceActiveFocus();
        }
    }
    Timer {
        id: queryTimer
        interval: 180
        onTriggered: {
            if (!app.indexing && !app.closing && (app.serviceStarting || (app.ready && app.status.modelsPresent)))
                app.search(app.query);
        }
    }
    // Quickshell selects the concrete panel type at startup; Qt's metadata
    // describes the abstract interface instead.
    // qmllint disable uncreatable-type
    PanelWindow {
        // qmllint enable uncreatable-type
        id: window
        property bool animateGeometry: false
        property bool geometryReady: false
        property bool preparingGeometry: false
        property bool surfaceReady: false
        // Keep the native rendering envelope stable. Moving/resizing it also
        // invokes compositor animation, which otherwise moves the card twice.
        // Input and compositor-visible masks cover only the painted card.
        readonly property int surfaceWidth: Math.min(Style.space(980), screen ? screen.width - Style.space(32) : Style.space(980))
        readonly property int surfaceHeight: Math.min(Style.space(740), screen ? screen.height - Style.space(32) : Style.space(740))
        property bool exclusiveKeyboard: true
        readonly property int compactHeight: Math.ceil(Style.space(56) + hotkeysFooter.implicitHeight)
        readonly property bool hasStatus: app.errorMessage.length > 0 || app.notice.length > 0 || app.showIndexReport || (app.ready && !app.status.modelsPresent) || app.indexing || app.installingModels
        readonly property bool expanded: app.page !== "summon" || hasStatus || errorDialog.visible || skippedFiles.visible || stopDialog.visible || restartDialog.visible
        readonly property int preferredWidth: Style.space(app.page === "results" ? 980 : app.page === "settings" ? 600 : 680)
        readonly property int preferredHeight: app.page === "settings" ? Style.space(640) : app.page === "results" || errorDialog.visible || skippedFiles.visible ? Style.space(640) : (stopDialog.visible || restartDialog.visible) ? Style.space(320) : hasStatus ? Math.ceil(compactHeight + statusArea.implicitHeight + Style.space(32)) : compactHeight
        readonly property int targetWidth: Math.min(panel.testWidth || preferredWidth, surfaceWidth)
        readonly property int targetHeight: Math.min(panel.testHeight || preferredHeight, surfaceHeight)
        implicitWidth: surfaceWidth
        implicitHeight: surfaceHeight
        exclusionMode: ExclusionMode.Ignore
        WlrLayershell.namespace: "fractal-search"
        // Native pickers and external viewers need to sit above this panel.
        WlrLayershell.layer: imageFolder.visible || modelFolder.visible || !exclusiveKeyboard ? WlrLayer.Bottom : WlrLayer.Overlay
        WlrLayershell.keyboardFocus: imageFolder.visible || modelFolder.visible ? WlrKeyboardFocus.None : WlrKeyboardFocus.OnDemand
        color: "transparent"
        surfaceFormat.opaque: false
        mask: Region {
            id: panelRegion
            item: canvas
        }
        // The input mask alone does not clip compositor effects or stale edges.
        // Quickshell's attached-type metadata omits Region here; native tests
        // validate this documented property against the compositor protocol.
        // qmllint disable missing-type
        HyprlandWindow.visibleMask: panelRegion
        // qmllint enable missing-type
        visible: false
        onClosed: app.dismiss()
        onTargetWidthChanged: if (geometryReady)
            geometryTimer.restart()
        onTargetHeightChanged: if (geometryReady)
            geometryTimer.restart()
        onVisibleChanged: {
            if (!visible)
                surfaceReady = false;
            if (geometryReady)
                geometryTimer.restart();
        }
        // Destroy this queued update with the panel during a host hot reload.
        // Qt.callLater can retain a bound function after its QML context dies.
        Timer {
            id: geometryTimer
            interval: 0
            onTriggered: window.updateGeometry()
        }
        function updateGeometry() {
            if (!geometryReady)
                return;
            resizeAnimation.stop();
            preparingGeometry = false;
            if (!animateGeometry || !visible || !surfaceReady || (canvas.width === targetWidth && canvas.height === targetHeight)) {
                finishGeometry();
                return;
            }
            preparingGeometry = true;
            canvas.Window.window.update();
        }
        function startPreparedAnimation() {
            if (!preparingGeometry || !visible || width !== surfaceWidth || height !== surfaceHeight)
                return;
            preparingGeometry = false;
            widthAnimation.from = canvas.width;
            widthAnimation.to = targetWidth;
            heightAnimation.from = canvas.height;
            heightAnimation.to = targetHeight;
            resizeAnimation.start();
        }
        function finishGeometry() {
            preparingGeometry = false;
            canvas.width = targetWidth;
            canvas.height = targetHeight;
        }
        Component.onCompleted: {
            finishGeometry();
            geometryReady = true;
            animateGeometry = true;
            if (visible)
                panel.focusSearch();
        }
        Connections {
            target: canvas.Window.window
            // Start after a correctly sized native buffer has been submitted.
            // Animating before the Wayland configure clips and jumps the card.
            function onFrameSwapped(): void {
                if (window.visible && window.width === window.surfaceWidth && window.height === window.surfaceHeight)
                    window.surfaceReady = true;
                window.startPreparedAnimation();
            }
        }
        ParallelAnimation {
            id: resizeAnimation
            onFinished: window.finishGeometry()
            NumberAnimation {
                id: widthAnimation
                target: canvas
                property: "width"
                duration: 160
                easing.type: Easing.OutCubic
            }
            NumberAnimation {
                id: heightAnimation
                target: canvas
                property: "height"
                duration: 160
                easing.type: Easing.OutCubic
            }
        }

        Rectangle {
            id: canvas
            // A new/remapped Qt window has a provisional size until Wayland
            // configures it. Present the card only once that buffer is ready.
            visible: window.surfaceReady
            onVisibleChanged: if (visible && !skippedFiles.visible && !errorDialog.visible && !stopDialog.visible && !restartDialog.visible)
                panel.focusSearch()
            anchors.centerIn: parent
            width: 0
            height: 0
            clip: true
            color: Qt.alpha(app.colors.background, 0.97)
            Rectangle {
                // The compositor's integer region can clip the outermost pixel
                // at fractional scale. Keep the outline inside the masked edge.
                anchors.fill: parent
                anchors.margins: 1
                z: 1
                color: "transparent"
                border.width: 1
                border.color: app.borderColor
            }
            // Choose the destination layout once, rather than switching rows
            // and preview visibility as the animated panel crosses breakpoints.
            readonly property bool narrow: window.targetWidth < 660
            readonly property bool shortWindow: window.targetHeight < 450
            Shortcut {
                sequence: "Ctrl+Q"
                enabled: !app.closing && !imageFolder.visible && !modelFolder.visible && !restartDialog.visible
                onActivated: panel.requestStop()
            }
            Shortcut {
                sequence: "Ctrl+,"
                enabled: app.page !== "settings" && !skippedFiles.visible && !errorDialog.visible && !removeDialog.visible && !stopDialog.visible && !restartDialog.visible
                onActivated: app.showSettings()
            }
            Shortcut {
                sequence: "Escape"
                enabled: !skippedFiles.visible && !errorDialog.visible && !removeDialog.visible && !stopDialog.visible && !restartDialog.visible && !imageFolder.visible && !modelFolder.visible
                onActivated: app.back()
            }
            Shortcut {
                sequence: "Ctrl+Return"
                enabled: app.page === "results" && !skippedFiles.visible && !stopDialog.visible && !restartDialog.visible
                onActivated: app.revealOriginal()
            }
            Shortcut {
                sequence: "Ctrl+Enter"
                enabled: app.page === "results" && !skippedFiles.visible && !stopDialog.visible && !restartDialog.visible
                onActivated: app.revealOriginal()
            }

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 1
                spacing: 0
                Item {
                    Layout.fillWidth: true
                    Layout.minimumHeight: 54
                    Layout.maximumHeight: 54
                    Layout.preferredHeight: 54
                    ColumnLayout {
                        anchors.centerIn: parent
                        width: parent.width
                        spacing: 0
                        Rectangle {
                            Layout.fillWidth: true
                            implicitHeight: Style.space(54)
                            color: "transparent"
                            RowLayout {
                                anchors.fill: parent
                                anchors.leftMargin: Style.space(16)
                                anchors.rightMargin: Style.space(12)
                                spacing: Style.space(12)
                                SearchButton {
                                    visible: app.page === "settings"
                                    text: "←"
                                    quiet: true
                                    colors: app.colors
                                    onClicked: app.back()
                                }
                                Item {
                                    visible: app.page !== "settings"
                                    implicitWidth: Style.space(18)
                                    implicitHeight: Style.space(20)
                                    Rectangle {
                                        width: 13
                                        height: 13
                                        radius: 7
                                        color: "transparent"
                                        border.color: app.colors.foreground
                                        border.width: 1
                                        y: 1
                                    }
                                    Rectangle {
                                        width: 7
                                        height: 1
                                        rotation: 45
                                        color: app.colors.foreground
                                        x: 11
                                        y: 15
                                    }
                                }
                                SearchText {
                                    visible: app.page === "settings"
                                    Layout.fillWidth: true
                                    text: "Settings"
                                    color: app.colors.foreground
                                    font.pixelSize: SearchTheme.font.subtitle
                                }
                                TextField {
                                    id: searchField
                                    visible: app.page !== "settings"
                                    Layout.fillWidth: true
                                    Layout.minimumWidth: 0
                                    padding: 0
                                    text: app.query
                                    placeholderText: app.serviceStarting && !app.errorMessage.length ? app.loadingMessage : "Describe what you’re looking for…"
                                    placeholderTextColor: app.secondaryText
                                    color: app.colors.foreground
                                    font.family: app.monoFont
                                    font.pixelSize: SearchTheme.font.title
                                    selectByMouse: true
                                    enabled: !app.closing
                                    background: null
                                    onTextEdited: {
                                        app.editQuery(text);
                                        if (text.trim().length)
                                            queryTimer.restart();
                                        else {
                                            queryTimer.stop();
                                            app.page = "summon";
                                        }
                                    }
                                    onAccepted: panel.submit()
                                    Keys.onDownPressed: event => {
                                        panel.navigate(1);
                                        event.accepted = true;
                                    }
                                    Keys.onUpPressed: event => {
                                        panel.navigate(-1);
                                        event.accepted = true;
                                    }
                                    Component.onCompleted: forceActiveFocus()
                                }
                                SearchButton {
                                    text: "⚙"
                                    font.pixelSize: 19
                                    Accessible.name: "Settings"
                                    visible: app.page !== "settings"
                                    colors: app.colors
                                    quiet: true
                                    onClicked: app.showSettings()
                                }
                                SearchButton {
                                    text: "Esc"
                                    implicitHeight: Style.space(24)
                                    font.pixelSize: SearchTheme.font.bodySmall
                                    visible: app.page === "results"
                                    colors: app.colors
                                    onClicked: app.back()
                                }
                            }
                        }
                    }
                }
                Item {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 0
                    // Keep the shrinking viewport in the layout until it closes,
                    // so the search bar and footer stay at their respective edges.
                    visible: window.expanded || canvas.height > window.compactHeight
                    clip: true
                    opacity: Math.min(1, Math.max(0, (canvas.height - window.compactHeight) / 160))
                    ColumnLayout {
                        // Reveal a settled layout through the growing viewport;
                        // resizing its images/text every frame makes expansion uneven.
                        width: Math.max(0, window.targetWidth - 2)
                        height: Math.max(0, window.targetHeight - 56 - hotkeysFooter.implicitHeight)
                        spacing: 0
                        Rectangle {
                            Layout.fillWidth: true
                            implicitHeight: 1
                            color: app.borderColor
                            visible: app.page !== "summon"
                        }

                        ColumnLayout {
                            id: statusArea
                            Layout.fillWidth: true
                            Layout.leftMargin: Style.space(16)
                            Layout.rightMargin: Style.space(16)
                            Layout.topMargin: visible ? 10 : 0
                            Layout.bottomMargin: visible ? 10 : 0
                            spacing: Style.space(7)
                            visible: window.hasStatus
                            RowLayout {
                                visible: !app.ready || (app.ready && !app.status.modelsPresent)
                                Layout.fillWidth: true
                                SearchText {
                                    Layout.fillWidth: true
                                    Layout.minimumWidth: 0
                                    text: !app.ready ? app.serviceStarting ? app.loadingMessage : "Search service unavailable" : "Set up the search model (~2.96 GB). Existing files are reused."
                                    color: app.secondaryText
                                    wrapMode: Text.WordWrap
                                }
                                SearchButton {
                                    text: "Set up model"
                                    colors: app.colors
                                    visible: app.ready && !app.status.modelsPresent
                                    enabled: !app.busy
                                    onClicked: app.installModels()
                                }
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                visible: !app.ready && !app.serviceStarting
                                SearchButton {
                                    text: "Retry service"
                                    colors: app.colors
                                    onClicked: app.ensureService()
                                }
                            }
                            ColumnLayout {
                                visible: app.installingModels
                                Layout.fillWidth: true
                                SearchText {
                                    text: (app.modelProgress.stage === "downloading" ? "Downloading " : "Verifying ") + app.modelProgress.filename
                                    color: app.colors.foreground
                                }
                                RowLayout {
                                    Layout.fillWidth: true
                                    ProgressBar {
                                        Layout.fillWidth: true
                                        from: 0
                                        to: app.modelProgress.total
                                        value: app.modelProgress.current
                                    }
                                    SearchButton {
                                        text: "Stop"
                                        colors: app.colors
                                        enabled: !app.closing
                                        onClicked: panel.requestStop()
                                    }
                                }
                            }
                            RowLayout {
                                visible: app.errorMessage.length > 0 || app.showIndexReport
                                Layout.fillWidth: true
                                SearchText {
                                    Layout.fillWidth: true
                                    Layout.minimumWidth: 0
                                    text: app.errorMessage
                                    color: app.colors.error
                                    wrapMode: Text.WordWrap
                                    maximumLineCount: canvas.shortWindow ? 2 : 3
                                    elide: Text.ElideRight
                                }
                                SearchButton {
                                    text: "Details"
                                    colors: app.colors
                                    visible: app.errorMessage.length > 0
                                    onClicked: errorDialog.open()
                                }
                                SearchButton {
                                    text: "Skipped files (" + app.indexFailureCount + ")"
                                    colors: app.colors
                                    visible: app.showIndexReport
                                    onClicked: skippedFiles.open()
                                }
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                visible: !app.indexing && (app.notice.length > 0 || app.errorMessage.length > 0 || app.showIndexReport)
                                SearchText {
                                    Layout.fillWidth: true
                                    Layout.minimumWidth: 0
                                    text: app.notice
                                    color: app.showIndexReport ? app.colors.error : app.colors.accent
                                    wrapMode: Text.WordWrap
                                    maximumLineCount: 3
                                    elide: Text.ElideRight
                                }
                                SearchButton {
                                    text: "Dismiss"
                                    colors: app.colors
                                    enabled: !app.busy
                                    onClicked: {
                                        app.acknowledgeStatus();
                                        panel.focusSearch();
                                    }
                                }
                            }
                            RowLayout {
                                visible: app.indexing
                                Layout.fillWidth: true
                                SearchText {
                                    text: app.progress.stage === "scanning" ? "Scanning folder…" : "Indexing " + app.progress.current + " / " + app.progress.total
                                    color: app.colors.foreground
                                }
                                SearchText {
                                    Layout.fillWidth: true
                                    Layout.minimumWidth: 0
                                    text: app.progress.filename
                                    color: app.secondaryText
                                    elide: Text.ElideMiddle
                                    font.pixelSize: SearchTheme.font.bodySmall
                                }
                                SearchButton {
                                    text: "Stop"
                                    colors: app.colors
                                    enabled: !app.closing
                                    onClicked: app.cancelIndex()
                                }
                            }
                            ProgressBar {
                                Layout.fillWidth: true
                                visible: app.indexing
                                from: 0
                                to: Math.max(1, app.progress.total)
                                value: app.progress.current
                                indeterminate: app.progress.stage === "scanning"
                                implicitHeight: Style.space(3)
                                palette.highlight: app.colors.accent
                            }
                        }

                        SettingsPage {
                            id: settingsPage
                            visible: app.page === "settings"
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            Layout.margins: canvas.shortWindow ? 16 : 24
                            Layout.maximumWidth: 660
                            Layout.alignment: Qt.AlignHCenter
                            app: app
                            onAddFolder: imageFolder.open()
                            onChooseModel: modelFolder.open()
                            onShowSkippedFiles: skippedFiles.open()
                            onStopService: panel.requestStop()
                            onRestartService: panel.requestRestart()
                            onRemoveFolder: (sourceId, directory) => {
                                removeDialog.sourceId = sourceId;
                                removeDialog.directory = directory;
                                removeDialog.open();
                            }
                        }

                        ColumnLayout {
                            visible: app.page === "results"
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            spacing: 0
                            RowLayout {
                                Layout.fillWidth: true
                                Layout.leftMargin: Style.space(16)
                                Layout.rightMargin: Style.space(12)
                                implicitHeight: Style.space(36)
                                SearchText {
                                    Layout.fillWidth: true
                                    text: app.loading ? app.loadingMessage : (app.activeQuery.length ? "Top " + app.results.count + " results" : app.results.count + " / " + app.total + " segments")
                                    color: app.secondaryText
                                }
                            }
                            RowLayout {
                                Layout.fillWidth: true
                                Layout.leftMargin: Style.space(16)
                                Layout.rightMargin: Style.space(12)
                                Layout.bottomMargin: Style.space(8)
                                spacing: Style.space(8)
                                SearchChoice {
                                    objectName: "mediaKindFilter"
                                    Accessible.name: "Media type"
                                    colors: app.colors
                                    Layout.fillWidth: true
                                    Layout.preferredWidth: 140
                                    Layout.minimumWidth: 100
                                    implicitHeight: Style.space(28)
                                    model: [
                                        { id: null, label: "All media" },
                                        { id: "image", label: "Images" },
                                        { id: "text", label: "Text" },
                                        { id: "audio", label: "Audio" },
                                        { id: "video", label: "Video" }
                                    ]
                                    textRole: "label"
                                    valueRole: "id"
                                    currentIndex: app.mediaKind === null ? 0 : indexOfValue(app.mediaKind)
                                    enabled: !app.indexing && !app.installingModels && !app.closing
                                    onActivated: {
                                        queryTimer.stop();
                                        app.setMediaKind(currentValue);
                                    }
                                }
                                SearchChoice {
                                    objectName: "sourceFilter"
                                    Accessible.name: "Folder"
                                    colors: app.colors
                                    Layout.fillWidth: true
                                    Layout.preferredWidth: 230
                                    Layout.minimumWidth: 100
                                    implicitHeight: Style.space(28)
                                    model: [
                                        {
                                            id: null,
                                            label: "All folders"
                                        }
                                    ].concat(app.status.catalog.sources.map(source => ({
                                                id: source.id,
                                                label: app.displayPath(source.directory)
                                            })))
                                    textRole: "label"
                                    valueRole: "id"
                                    currentIndex: app.sourceId === null ? 0 : indexOfValue(app.sourceId)
                                    enabled: !app.indexing && !app.installingModels && !app.closing
                                    onActivated: {
                                        queryTimer.stop();
                                        app.setSourceId(currentValue);
                                    }
                                }
                            }
                            Rectangle {
                                Layout.fillWidth: true
                                implicitHeight: 1
                                color: app.borderColor
                            }
                            Item {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                RowLayout {
                                    anchors.fill: parent
                                    spacing: 0
                                    ListView {
                                        id: resultsList
                                        Layout.fillHeight: true
                                        Layout.fillWidth: true
                                        Layout.preferredWidth: window.targetWidth * 0.42
                                        Layout.minimumWidth: 0
                                        clip: true
                                        model: app.results
                                        currentIndex: app.selectedIndex
                                        boundsBehavior: Flickable.StopAtBounds
                                        cacheBuffer: 250
                                        ScrollBar.vertical: ScrollBar {
                                            policy: ScrollBar.AsNeeded
                                        }
                                        Keys.onDownPressed: panel.navigate(1)
                                        Keys.onUpPressed: panel.navigate(-1)
                                        Keys.onReturnPressed: app.openOriginal()
                                        delegate: Rectangle {
                                            id: resultRow
                                            required property var model
                                            required property int index
                                            width: resultsList.width
                                            height: Math.max(canvas.shortWindow ? 70 : 83, resultSummary.implicitHeight + Style.space(18))
                                            color: index === app.selectedIndex ? Qt.alpha(app.colors.accent, 0.13) : rowMouse.containsMouse ? Qt.alpha(app.colors.foreground, 0.035) : "transparent"
                                            Rectangle {
                                                x: 8
                                                y: 9
                                                height: parent.height - 18
                                                width: 2
                                                color: app.colors.accent
                                                visible: resultRow.index === app.selectedIndex
                                            }
                                            RowLayout {
                                                anchors.fill: parent
                                                anchors.margins: Style.space(9)
                                                anchors.leftMargin: Style.space(18)
                                                anchors.rightMargin: Style.space(16)
                                                spacing: Style.space(14)
                                                Rectangle {
                                                    Layout.preferredWidth: canvas.shortWindow ? 76 : 94
                                                    Layout.fillHeight: true
                                                    color: Qt.alpha(app.colors.foreground, 0.05)
                                                    radius: 4
                                                    Image {
                                                        anchors.fill: parent
                                                        visible: app.isVisualAsset(resultRow.model)
                                                        source: visible ? resultRow.model.thumbnailUrl : ""
                                                        sourceSize: Qt.size(192, 160)
                                                        fillMode: Image.PreserveAspectCrop
                                                        asynchronous: true
                                                        onStatusChanged: if (status === Image.Error)
                                                            app.fail("Could not display thumbnail for " + resultRow.model.filename)
                                                    }
                                                    SearchText {
                                                        anchors.centerIn: parent
                                                        visible: !app.isVisualAsset(resultRow.model)
                                                        text: resultRow.model.mediaKind === "audio" ? "Audio" : "Text"
                                                        color: app.secondaryText
                                                        font.pixelSize: SearchTheme.font.bodySmall
                                                    }
                                                }
                                                ColumnLayout {
                                                    id: resultSummary
                                                    Layout.fillWidth: true
                                                    Layout.minimumWidth: 0
                                                    spacing: Style.space(3)
                                                    SearchText {
                                                        Layout.fillWidth: true
                                                        Layout.minimumWidth: 0
                                                        text: resultRow.model.filename
                                                        color: resultRow.index === app.selectedIndex ? app.colors.accent : app.colors.foreground
                                                        elide: Text.ElideMiddle
                                                        font.pixelSize: SearchTheme.font.subtitle
                                                    }
                                                    SearchText {
                                                        Layout.fillWidth: true
                                                        Layout.minimumWidth: 0
                                                        text: app.mediaLabel(resultRow.model) + " · " + (app.folderPath(resultRow.model.path).split("/").filter(Boolean).pop() || "/")
                                                        color: app.secondaryText
                                                        elide: Text.ElideMiddle
                                                        font.pixelSize: SearchTheme.font.bodySmall
                                                    }
                                                    SearchText {
                                                        Layout.fillWidth: true
                                                        Layout.minimumWidth: 0
                                                        visible: resultRow.model.hasHybridScore || resultRow.model.hasScore
                                                        text: app.scoreLabel(resultRow.model)
                                                        color: app.secondaryText
                                                        elide: Text.ElideRight
                                                        font.pixelSize: SearchTheme.font.bodySmall
                                                    }
                                                    SearchText {
                                                        Layout.fillWidth: true
                                                        Layout.minimumWidth: 0
                                                        visible: resultRow.model.textExcerpt.length > 0
                                                        text: resultRow.model.textExcerpt.replace(/\s+/g, " ")
                                                        color: app.secondaryText
                                                        elide: Text.ElideRight
                                                        font.pixelSize: SearchTheme.font.bodySmall
                                                    }
                                                }
                                            }
                                            MouseArea {
                                                id: rowMouse
                                                anchors.fill: parent
                                                hoverEnabled: true
                                                onClicked: {
                                                    app.select(resultRow.index);
                                                    searchField.forceActiveFocus();
                                                }
                                                onDoubleClicked: app.openOriginal()
                                            }
                                        }
                                    }
                                    Rectangle {
                                        Layout.fillHeight: true
                                        implicitWidth: 1
                                        color: app.borderColor
                                        visible: previewPane.visible
                                    }
                                    ColumnLayout {
                                        id: previewPane
                                        visible: app.results.count > 0 && app.settings.resultLayout === "listPreview" && !canvas.narrow
                                        Layout.fillWidth: true
                                        Layout.fillHeight: true
                                        Layout.preferredWidth: window.targetWidth * 0.58
                                        Layout.minimumWidth: 0
                                        Layout.margins: 16
                                        spacing: Style.space(10)
                                        Item {
                                            Layout.fillWidth: true
                                            Layout.fillHeight: true
                                            Image {
                                                anchors.fill: parent
                                                visible: app.isVisualAsset(app.selectedAsset)
                                                source: visible ? app.previewUrl : ""
                                                asynchronous: true
                                                fillMode: Image.PreserveAspectFit
                                                onStatusChanged: if (status === Image.Error) {
                                                    app.previewError = "The visual preview could not be displayed.";
                                                    app.fail(app.previewError);
                                                }
                                            }
                                            SearchText {
                                                anchors.centerIn: parent
                                                width: parent.width - 20
                                                horizontalAlignment: app.selectedAsset && app.selectedAsset.mediaKind === "text" ? Text.AlignLeft : Text.AlignHCenter
                                                text: app.previewError.length ? app.previewError : app.previewLoading && app.selectedAsset && app.selectedAsset.mediaKind === "image" ? "Loading preview…" : app.previewText()
                                                color: app.previewError.length ? app.colors.error : app.secondaryText
                                                wrapMode: Text.WordWrap
                                                maximumLineCount: Math.max(1, Math.floor((parent.height - 20) / font.pixelSize / 1.4))
                                                elide: Text.ElideRight
                                            }
                                        }
                                        RowLayout {
                                            Layout.fillWidth: true
                                            SearchText {
                                                Layout.fillWidth: true
                                                Layout.minimumWidth: 0
                                                text: app.selectedAsset ? app.selectedAsset.filename : ""
                                                color: app.colors.foreground
                                                elide: Text.ElideMiddle
                                                font.pixelSize: SearchTheme.font.subtitle
                                            }
                                            SearchText {
                                                text: app.selectedAsset ? app.mediaLabel(app.selectedAsset) : ""
                                                color: app.secondaryText
                                                font.pixelSize: SearchTheme.font.bodySmall
                                            }
                                        }
                                        SearchText {
                                            Layout.fillWidth: true
                                            visible: text.length > 0
                                            text: app.scoreLabel(app.selectedAsset)
                                            color: app.secondaryText
                                            font.pixelSize: SearchTheme.font.bodySmall
                                        }
                                        SearchText {
                                            Layout.fillWidth: true
                                            Layout.minimumWidth: 0
                                            text: app.selectedAsset ? app.displayPath(app.folderPath(app.selectedAsset.path)) : ""
                                            color: app.secondaryText
                                            elide: Text.ElideMiddle
                                            font.pixelSize: SearchTheme.font.bodySmall
                                        }
                                        ResultActions {
                                            Layout.fillWidth: true
                                            app: app
                                        }
                                    }
                                }
                                ColumnLayout {
                                    anchors.centerIn: parent
                                    width: Math.min(parent.width - 40, 350)
                                    visible: !app.results.count
                                    spacing: Style.space(12)
                                    SearchText {
                                        Layout.fillWidth: true
                                        text: app.serviceStarting || app.loading ? app.loadingMessage : app.indexing ? "Building your search index…" : app.errorMessage.length ? "Could not load results" : app.activeQuery.length ? "No matching results" : "Your library, searchable"
                                        color: app.colors.foreground
                                        font.pixelSize: 17
                                        horizontalAlignment: Text.AlignHCenter
                                        wrapMode: Text.WordWrap
                                    }
                                    SearchText {
                                        Layout.fillWidth: true
                                        visible: !app.loading && !app.indexing
                                        text: app.errorMessage.length ? "Resolve the error above, then try again." : app.activeQuery.length ? "Try another description or choose a different folder." : "Add a folder in settings to search images, text, audio and video locally."
                                        color: app.secondaryText
                                        horizontalAlignment: Text.AlignHCenter
                                        wrapMode: Text.WordWrap
                                    }
                                    SearchButton {
                                        Layout.alignment: Qt.AlignHCenter
                                        visible: !app.loading && !app.indexing && !app.activeQuery.length
                                        text: "+ Add folder"
                                        colors: app.colors
                                        enabled: app.ready && app.status.modelsPresent && !app.busy
                                        onClicked: imageFolder.open()
                                    }
                                }
                            }
                            ResultActions {
                                visible: app.results.count > 0 && !previewPane.visible
                                Layout.fillWidth: true
                                Layout.margins: Style.space(12)
                                app: app
                            }
                            SearchButton {
                                Layout.alignment: Qt.AlignHCenter
                                Layout.margins: 6
                                visible: app.results.count < app.total && !app.activeQuery.length
                                text: "Load more"
                                colors: app.colors
                                enabled: app.ready && !app.busy
                                onClicked: app.browse(true)
                            }
                        }
                    }
                }
                ColumnLayout {
                    id: hotkeysFooter
                    Layout.fillWidth: true
                    spacing: 0
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: 1
                        color: app.borderColor
                    }
                    Item {
                        Layout.fillWidth: true
                        Layout.leftMargin: Style.space(10)
                        Layout.rightMargin: Style.space(10)
                        Layout.topMargin: Style.space(6)
                        Layout.bottomMargin: Style.space(6)
                        implicitHeight: hotkeysFlow.implicitHeight
                        Flow {
                            id: hotkeysFlow
                            width: Math.max(0, window.targetWidth - 22)
                            spacing: Style.space(8)
                            Row {
                                visible: app.page === "results"
                                height: 26
                                spacing: Style.space(6)
                                ShortcutKey {
                                    anchors.verticalCenter: parent.verticalCenter
                                    text: "↑ ↓"
                                    colors: app.colors
                                }
                                SearchText {
                                    text: "Navigate"
                                    color: app.secondaryText
                                    font.pixelSize: SearchTheme.font.bodySmall
                                    height: 26
                                }
                            }
                            FooterButton {
                                visible: app.page !== "settings"
                                text: app.page === "results" ? "Open" : "Search"
                                shortcut: "Enter"
                                colors: app.colors
                                enabled: app.ready && !app.busy && !app.closing && (app.page !== "results" || !!app.selectedAsset)
                                onClicked: panel.submit()
                            }
                            FooterButton {
                                visible: app.page === "summon"
                                text: "Settings"
                                shortcut: "Ctrl+,"
                                colors: app.colors
                                onClicked: app.showSettings()
                            }
                            FooterButton {
                                visible: app.page !== "summon"
                                text: "Back"
                                shortcut: "Esc"
                                colors: app.colors
                                enabled: !app.closing
                                onClicked: app.back()
                            }
                            FooterButton {
                                visible: panel.closeShortcutEnabled
                                text: "Close"
                                shortcut: "Super+W"
                                colors: app.colors
                                onClicked: app.dismiss()
                            }
                        }
                    }
                }
            }

            PortalFolderDialog {
                id: imageFolder
                executable: panel.pickerExecutable
                title: "Choose a folder to index"
                onVisibleChanged: if (!visible && window.visible)
                    panel.focusSearch()
                onAccepted: app.indexFolder(selectedFolder.toString())
                onFailed: message => app.fail(message)
            }
            PortalFolderDialog {
                id: modelFolder
                executable: panel.pickerExecutable
                title: "Choose the downloaded EmbeddingGemma 2 model folder"
                onVisibleChanged: if (!visible && window.visible)
                    panel.focusSearch()
                onAccepted: app.setModelDirectory(selectedFolder.toString())
                onFailed: message => app.fail(message)
            }
            SearchConfirmationDialog {
                id: restartDialog
                parent: canvas
                colors: app.colors
                borderColor: app.borderColor
                title: "Restart Fractal Search?"
                message: "Image batch size is saved as " + app.settings.imageBatchSize + ". Restart the search service to apply it.\n\nActive inference will finish and completed files will be saved before restart."
                confirmText: "Restart now"
                cancelText: "Later"
                onClosed: panel.focusSearch()
                onAccepted: app.restartService()
            }
            SearchConfirmationDialog {
                id: stopDialog
                parent: canvas
                colors: app.colors
                borderColor: app.borderColor
                title: "Stop search service?"
                message: "Indexing and folder watching will stop. Active work finishes safely and indexed files are saved.\n\nOpen Fractal Search again to restart the service."
                confirmText: "Stop service"
                onClosed: panel.focusSearch()
                onAccepted: app.shutdown()
            }
            SearchConfirmationDialog {
                id: removeDialog
                property int sourceId: -1
                property string directory: ""
                // Keep popups on the persistent canvas across hide/summon.
                parent: canvas
                colors: app.colors
                borderColor: app.borderColor
                title: "Remove indexed folder?"
                message: app.displayPath(directory) + "\n\nRemove its searchable segments from the index. Original files stay in place."
                confirmText: "Remove folder"
                onClosed: panel.focusSearch()
                onAccepted: app.removeSource(sourceId)
            }
            Dialog {
                id: skippedFiles
                onClosed: panel.focusSearch()
                parent: canvas
                anchors.centerIn: parent
                width: Math.min(680, canvas.width - 24)
                height: Math.min(560, canvas.height - 24)
                modal: true
                popupType: Popup.Item
                focus: true
                enter: Transition {}
                exit: Transition {}
                title: "Skipped files (" + app.indexFailureCount + ")"
                palette.windowText: app.colors.foreground
                palette.text: app.colors.foreground
                palette.buttonText: app.colors.foreground
                palette.button: app.colors.surface
                standardButtons: Dialog.Close
                background: Rectangle {
                    color: app.colors.background
                    border.color: app.borderColor
                }
                contentItem: ColumnLayout {
                    SearchText {
                        Layout.fillWidth: true
                        text: "Repair these files and rescan the folder." + (app.indexFailures.length < app.indexFailureCount ? " Showing the first " + app.indexFailures.length + " errors." : "")
                        color: app.colors.foreground
                        wrapMode: Text.WordWrap
                    }
                    ScrollView {
                        id: skippedScroll
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        contentWidth: availableWidth
                        clip: true
                        TextArea {
                            width: skippedScroll.availableWidth
                            text: app.indexFailureDetails
                            textFormat: TextEdit.PlainText
                            wrapMode: TextEdit.Wrap
                            readOnly: true
                            selectByMouse: true
                            color: app.colors.foreground
                            font.family: app.monoFont
                            font.pixelSize: SearchTheme.font.body
                            background: null
                        }
                    }
                }
            }
            Dialog {
                id: errorDialog
                onClosed: panel.focusSearch()
                parent: canvas
                anchors.centerIn: parent
                width: Math.min(680, canvas.width - 24)
                height: Math.min(400, canvas.height - 24)
                modal: true
                popupType: Popup.Item
                focus: true
                enter: Transition {}
                exit: Transition {}
                title: "Search error"
                palette.windowText: app.colors.foreground
                palette.buttonText: app.colors.foreground
                palette.button: app.colors.surface
                standardButtons: Dialog.Close
                background: Rectangle {
                    color: app.colors.background
                    border.color: app.borderColor
                }
                contentItem: ScrollView {
                    id: errorScroll
                    contentWidth: availableWidth
                    clip: true
                    TextArea {
                        width: errorScroll.availableWidth
                        text: app.errorMessage
                        readOnly: true
                        selectByMouse: true
                        wrapMode: TextEdit.Wrap
                        color: app.colors.error
                        font.family: app.monoFont
                        font.pixelSize: SearchTheme.font.body
                        background: null
                    }
                }
            }
        }
    }
    // Omarchy calls these lifecycle methods through its scoped plugin facade.
    function open(payloadJson) {
        try {
            const payload = payloadJson ? JSON.parse(payloadJson) : {};
            if (!payload || typeof payload !== "object" || Array.isArray(payload) || Object.keys(payload).some(key => key !== "query") || (payload.query !== undefined && typeof payload.query !== "string"))
                throw new Error("Expected an object with an optional query string.");
            show();
            if (payload.query !== undefined)
                app.search(payload.query);
        } catch (error) {
            app.fail("Cannot open Fractal Search: " + error);
            show();
        }
    }
    function ping() {
        return runtimeConfig.bundleVersion;
    }
    function close() {
        app.dismiss();
    }
    function query(text) {
        show();
        app.search(text);
    }
    function quit() {
        app.shutdown();
    }
    function requestStop() {
        if (app.closing || stopDialog.visible)
            return;
        queryTimer.stop();
        skippedFiles.close();
        errorDialog.close();
        removeDialog.close();
        restartDialog.close();
        stopDialog.open();
    }
    function requestRestart() {
        if (app.closing || !app.batchRestartRequired || restartDialog.visible)
            return;
        queryTimer.stop();
        skippedFiles.close();
        errorDialog.close();
        removeDialog.close();
        stopDialog.close();
        restartDialog.open();
    }
    // The desktop close binding must never close the app behind our overlay.
    // Native pickers release panel focus; cancel their request only when the
    // compositor reports that our picker is the active application window.
    function closeFocused(payloadJson) {
        const payload = JSON.parse(payloadJson);
        if (!payload || typeof payload !== "object" || Array.isArray(payload) || Object.keys(payload).some(key => key !== "title" && key !== "address") || typeof payload.title !== "string" || typeof payload.address !== "string")
            throw new Error("Expected the active window title and address.");
        if (imageFolder.visible && payload.address && payload.title === imageFolder.title) {
            imageFolder.close();
            return "handled";
        }
        if (modelFolder.visible && payload.address && payload.title === modelFolder.title) {
            modelFolder.close();
            return "handled";
        }
        if (!window.visible || !canvas.Window.active || !window.exclusiveKeyboard || imageFolder.visible || modelFolder.visible)
            return "pass";
        app.dismiss();
        return "handled";
    }
    signal stopped
    Connections {
        target: app
        function onStopped(): void {
            panel.stopped();
        }
    }
    IpcHandler {
        enabled: Quickshell.env("FRACTAL_SEARCH_UI_TEST") === "1"
        target: "test"
        function index(directory: string): void {
            app.indexFolder(directory);
        }
        function state(): string {
            return JSON.stringify({
                ready: app.ready,
                installingModels: app.installingModels,
                modelsPresent: app.status.modelsPresent,
                modelDirectory: app.status.modelDirectory,
                query: app.query,
                activeQuery: app.activeQuery,
                sourceId: app.sourceId,
                loading: app.loading,
                serviceStarting: app.serviceStarting,
                searchStarting: app.searchStarting,
                loadingMessage: app.loadingMessage,
                previewLoading: app.previewLoading,
                indexing: app.indexing,
                error: app.errorMessage,
                notice: app.notice,
                failed: app.indexFailureCount,
                runtimeDiagnosticChecking: app.runtimeDiagnostics.checking,
                runtimeDiagnosticSummary: app.runtimeDiagnostics.summary,
                runtimeDiagnosticError: app.runtimeDiagnostics.errorMessage,
                runtimeDiagnosticDetails: app.runtimeDiagnostics.details,
                indexReportAcknowledged: app.indexReportAcknowledged,
                failures: app.indexFailures,
                skippedFilesVisible: skippedFiles.visible,
                count: app.results.count,
                total: app.total,
                assetCount: app.status.catalog.assetCount,
                fileCount: app.status.catalog.fileCount,
                progress: app.progress,
                previewError: app.previewError,
                background: app.colors.background,
                width: window.width,
                height: window.height,
                surfaceWidth: window.surfaceWidth,
                surfaceHeight: window.surfaceHeight,
                panelWidth: canvas.width,
                panelHeight: canvas.height,
                panelVisible: canvas.visible,
                targetWidth: window.targetWidth,
                targetHeight: window.targetHeight,
                compactHeight: window.compactHeight,
                hotkeysVisible: hotkeysFooter.visible,
                hotkeysHeight: hotkeysFooter.height,
                animating: window.preparingGeometry || resizeAnimation.running,
                active: canvas.Window.window ? canvas.Window.window.active : false,
                folderVisible: imageFolder.visible,
                modelFolderVisible: modelFolder.visible,
                stopConfirmationVisible: stopDialog.visible,
                stopCancelFocused: stopDialog.cancelFocused,
                restartConfirmationVisible: restartDialog.visible,
                restartLaterFocused: restartDialog.cancelFocused,
                restarting: app.restarting,
                activeImageBatchSize: app.status.activeImageBatchSize,
                batchRestartRequired: app.batchRestartRequired,
                stopConfirmGeometry: {
                    x: canvas.x + stopDialog.confirmButton.mapToItem(canvas, 0, 0).x,
                    y: canvas.y + stopDialog.confirmButton.mapToItem(canvas, 0, 0).y,
                    width: stopDialog.confirmButton.width,
                    height: stopDialog.confirmButton.height
                },
                closing: app.closing,
                closeShortcutEnabled: panel.closeShortcutEnabled,
                previewUrl: app.previewUrl,
                page: app.page,
                settingsTab: settingsPage.currentTab,
                selectedIndex: app.selectedIndex,
                selectedAsset: app.selectedAsset,
                settings: app.settings,
                visible: window.visible,
                sources: app.status.catalog.sources
            });
        }
        function preview(index: int): void {
            app.page = "results";
            app.select(index);
        }
        function reveal(): void {
            app.revealOriginal();
        }
        function openOriginal(): void {
            app.openOriginal();
        }
        function model(directory: string): void {
            app.setModelDirectory(directory);
        }
        function source(sourceId: int): void {
            app.sourceId = sourceId < 0 ? null : sourceId;
            app.browse(false);
        }
        function remove(sourceId: int): void {
            const source = app.status.catalog.sources.find(source => source.id === sourceId);
            if (!source) {
                app.fail("Test source does not exist.");
                return;
            }
            removeDialog.sourceId = sourceId;
            removeDialog.directory = source.directory;
            removeDialog.open();
        }
        function confirmRemove(): void {
            removeDialog.accept();
        }
        function reindex(): void {
            app.reindexAll();
        }
        function navigate(delta: int): void {
            panel.navigate(delta);
        }
        function settings(): void {
            app.showSettings();
        }
        function settingsTab(index: int): void {
            settingsPage.selectTab(index);
        }
        function refreshDiagnostics(): void {
            app.runtimeDiagnostics.refresh();
        }
        function updateSettings(json: string): void {
            app.updateSettings(JSON.parse(json));
        }
        function browse(): void {
            app.search("");
        }
        function back(): void {
            app.back();
        }
        function resize(width: int, height: int): void {
            panel.testWidth = width;
            panel.testHeight = height;
        }
        function cancel(): void {
            app.cancelIndex();
        }
        function closeWindow(): void {
            canvas.Window.window.close();
        }
        function closePreview(): void {
            app.selectedAsset = null;
            app.previewUrl = "";
        }
        function openSkippedFiles(): void {
            skippedFiles.open();
        }
        function closeSkippedFiles(): void {
            skippedFiles.close();
        }
        function acknowledge(): void {
            app.acknowledgeStatus();
        }
        function setupModels(): void {
            app.installModels();
        }
        function openError(): void {
            errorDialog.open();
        }
        function closeError(): void {
            errorDialog.close();
        }
        function openFolder(): void {
            imageFolder.open();
        }
        function closeFolder(): void {
            imageFolder.close();
        }
        function openModelFolder(): void {
            modelFolder.open();
        }
        function closeModelFolder(): void {
            modelFolder.close();
        }
        function requestStop(): void {
            panel.requestStop();
        }
        function confirmStop(): void {
            stopDialog.accept();
        }
        function confirmRestart(): void {
            restartDialog.accept();
        }
        function deferRestart(): void {
            restartDialog.reject();
        }
        function requestRestart(): void {
            panel.requestRestart();
        }
        function manageTestService(enabled: bool): void {
            panel.manageService = enabled;
        }
        function closeFocused(payload: string): string {
            return panel.closeFocused(payload);
        }
        function screenshot(path: string): void {
            if (!path.startsWith("/tmp/fractal-search-")) {
                app.fail("Test screenshot path must be under /tmp/fractal-search-");
                return;
            }
            const target = skippedFiles.visible ? skippedFiles.contentItem : canvas;
            if (!target.grabToImage(function (result) {
                if (!result.saveToFile(path))
                    app.fail("Cannot save test screenshot.");
            }))
                app.fail("Could not start test screenshot capture.");
        }
    }
}
