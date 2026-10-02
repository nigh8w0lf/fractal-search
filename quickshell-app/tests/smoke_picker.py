#!/usr/bin/env python3
"""Portal protocol checks on a private session bus; no desktop or user data."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from gi.repository import Gio, GLib

PORTAL = 'org.freedesktop.portal.Desktop'
ROOT = '/org/freedesktop/portal/desktop'
CHOOSER = 'org.freedesktop.portal.FileChooser'
REQUEST = 'org.freedesktop.portal.Request'
XML = f'''<node>
<interface name="{CHOOSER}">
 <method name="OpenFile"><arg type="s" direction="in"/><arg type="s" direction="in"/>
 <arg type="a{{sv}}" direction="in"/><arg type="o" direction="out"/></method>
 <property name="version" type="u" access="read"/>
</interface>
<interface name="{REQUEST}">
 <method name="Close"/><signal name="Response"><arg type="u"/><arg type="a{{sv}}"/></signal>
</interface>
</node>'''


def main():
    binary = str(Path(sys.argv[1]).resolve())
    connection = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    connection.call_sync('org.freedesktop.DBus', '/org/freedesktop/DBus',
                         'org.freedesktop.DBus', 'RequestName',
                         GLib.Variant('(su)', (PORTAL, 0)), None,
                         Gio.DBusCallFlags.NONE, 5000, None)
    info = Gio.DBusNodeInfo.new_for_xml(XML)
    scenario, requests, closed, errors = {}, [], [], []

    def close(conn, sender, path, interface, method, parameters, invocation):
        closed.append(path)
        invocation.return_value(GLib.Variant('()', ()))

    def open_folder(conn, sender, path, interface, method, parameters, invocation):
        try:
            parent, title, options = parameters.unpack()
            assert parent == '' and title == 'Picker protocol test'
            assert options['directory'] is True and options['multiple'] is False
            assert options['modal'] is False
            request = f"{ROOT}/request/{sender[1:].replace('.', '_')}/{options['handle_token']}"
            requests.append((request, conn.register_object(request, info.interfaces[1], close, None, None)))
            response = scenario.get('response')
            if response is not None:
                code, uris = response
                values = {} if uris is None else {'uris': GLib.Variant('as', uris)}
                # Emit before the method reply to test subscription order.
                conn.emit_signal(sender, request, REQUEST, 'Response', GLib.Variant('(ua{sv})', (code, values)))
            invocation.return_value(GLib.Variant('(o)', (request,)))
        except Exception as error:
            errors.append(error)
            invocation.return_dbus_error('org.freedesktop.portal.Error.Failed', str(error))

    chooser = connection.register_object(ROOT, info.interfaces[0], open_folder,
                                         lambda *_: GLib.Variant('u', scenario.get('version', 4)), None)

    def pump(predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while not predicate():
            assert time.monotonic() < deadline, 'Picker did not finish in time'
            while GLib.MainContext.default().pending():
                GLib.MainContext.default().iteration(False)
            time.sleep(.005)
        assert not errors, errors

    def run(response=None, cancel=False, version=4, disconnect=False):
        scenario.clear()
        scenario.update(response=response, version=version)
        before = len(requests)
        child = subprocess.Popen([binary, '--title', 'Picker protocol test'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            if cancel:
                pump(lambda: len(requests) > before)
                child.send_signal(signal.SIGTERM)
            if disconnect:
                pump(lambda: len(requests) > before)
                connection.call_sync('org.freedesktop.DBus', '/org/freedesktop/DBus', 'org.freedesktop.DBus', 'ReleaseName', GLib.Variant('(s)', (PORTAL,)), None, Gio.DBusCallFlags.NONE, 5000, None)
            pump(lambda: child.poll() is not None)
            output, diagnostics = child.communicate()
            value = json.loads(output)
            assert child.returncode == (0 if value['ok'] else 1), diagnostics
            return value
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    with tempfile.TemporaryDirectory(prefix='fractal-picker-', dir='/tmp') as temporary:
        directory = Path(temporary) / 'folder with spaces and ünicode'
        directory.mkdir()
        uri = directory.as_uri()
        assert run((0, [uri])) == {'ok': True, 'cancelled': False, 'folder': uri}
        assert run((1, None)) == {'ok': True, 'cancelled': True}
        assert run(cancel=True) == {'ok': True, 'cancelled': True}
        assert closed, 'SIGTERM did not close the portal request'
        for response in [(2, None), (0, None), (0, []), (0, [uri, uri]), (0, ['https://example.com']), (0, ['file:///no-such-fractal-folder'])]:
            assert not run(response)['ok'], response
        assert not run(version=2)['ok']
        assert not run(disconnect=True)['ok'], 'Portal exit should fail promptly'
        connection.call_sync('org.freedesktop.DBus', '/org/freedesktop/DBus', 'org.freedesktop.DBus', 'RequestName', GLib.Variant('(su)', (PORTAL, 0)), None, Gio.DBusCallFlags.NONE, 5000, None)
        connection.unregister_object(chooser)
        assert not run()['ok'], 'Missing portal interface should fail visibly'
    for _, registration in requests:
        connection.unregister_object(registration)
    print('Picker: early response, local Unicode selection, cancellation, SIGTERM Close, invalid responses, portal disconnect and old/missing portal passed')


if __name__ == '__main__':
    assert os.environ.get('FRACTAL_PICKER_PRIVATE_BUS') == '1', 'Run with FRACTAL_PICKER_PRIVATE_BUS=1 dbus-run-session -- python3 ...'
    main()
