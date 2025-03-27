"""
This file is part of Audio2Blender.

Audio2Blender is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Audio2Blender is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Audio2Blender. If not, see <https://www.gnu.org/licenses/>.
"""

bl_info = {
    "name": "Audio2Blender",
    "author": "Tom Mala Easy",  # Replace with your signature
    "version": "0.0.1",
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > Audio2Bl",
    "description": "Addon for audio sampling and analysis (Volume, FFT) updating a mesh in real time.",
    "warning": "",
    "wiki_url": "",
    "category": "Object",
}

import bpy
import numpy as np
import sounddevice as sd
import threading
import time
import platform

# Default settings
SAMPLERATE = 44100
DEFAULT_BLOCK_SIZE = 512
DEFAULT_FPS = 60.0

# Amplification factor to enhance the effect of gain in volume mode
AMPLIFICATION_FACTOR = 50

# Disable debug logging
DEBUG = False

# Global variables for data sharing
audio_stream = None
audio_config = {}  # Audio configuration at startup
latest_samples = None  # Samples captured from callback
latest_levels = None  # Analysis result (FFT/volume/bands)
data_lock = threading.Lock()

# Global precalculated FFT parameters
fft_params = None

# Variable to monitor last audio sample update
last_sample_time = time.time()

def precalc_fft_params(block_size, analysis_mode, num_bands=None, lower_bound=20, upper_bound=20000):
    freqs = np.fft.rfftfreq(block_size, d=1.0 / SAMPLERATE)
    if analysis_mode in ("NBANDS", "FFT_LOG") and num_bands is not None:
        if analysis_mode == "NBANDS":
            # Equally spaced bands (FFT Linear)
            band_width = (upper_bound - lower_bound) / num_bands
            bands = []
            for i in range(num_bands):
                lb = lower_bound + i * band_width
                ub = lower_bound + (i + 1) * band_width
                idx = (freqs >= lb) & (freqs < ub)
                bands.append(idx)
        else:  # FFT_LOG
            # Logarithmically spaced edges along the frequency range
            edges = np.geomspace(lower_bound, upper_bound, num=num_bands+1)
            bands = []
            for i in range(num_bands):
                lb = edges[i]
                ub = edges[i+1]
                idx = (freqs >= lb) & (freqs < ub)
                bands.append(idx)
        return {"freqs": freqs, "bands": bands,
                "spectrum_min": lower_bound, "spectrum_max": upper_bound}
    return {"freqs": freqs}

def create_audio_source_object(mode, num_bands=None):
    if mode == "VOLUME":
        obj_name = "Volume"
    elif mode in ("NBANDS", "FFT_LOG"):
        obj_name = "FFT"
    else:
        obj_name = "AudioSource"
    
    if obj_name in bpy.data.objects:
        obj = bpy.data.objects[obj_name]
        mesh = obj.data
        if mode in ("NBANDS", "FFT_LOG"):
            if len(mesh.vertices) != num_bands:
                new_mesh = bpy.data.meshes.new(obj_name + "_Mesh")
                new_mesh.from_pydata([(float(i), 0, 0) for i in range(num_bands)], [], [])
                new_mesh.update()
                if "fft" not in new_mesh.attributes:
                    attr = new_mesh.attributes.new(name="fft", type='FLOAT', domain='POINT')
                    for i in range(num_bands):
                        attr.data[i].value = 0.0
                obj.data = new_mesh
            else:
                if "fft" not in mesh.attributes:
                    attr = mesh.attributes.new(name="fft", type='FLOAT', domain='POINT')
                    for i in range(num_bands):
                        attr.data[i].value = 0.0
        elif mode == "VOLUME":
            if "volume" not in mesh.attributes:
                mesh.attributes.new(name="volume", type='FLOAT', domain='POINT')
                mesh.attributes["volume"].data[0].value = 0.0
        if DEBUG:
            print(f"create_audio_source_object: Reusing existing object {obj_name}")
        return obj
    else:
        if mode in ("NBANDS", "FFT_LOG"):
            vertices = [(float(i), 0, 0) for i in range(num_bands)]
            mesh = bpy.data.meshes.new(obj_name + "_Mesh")
            mesh.from_pydata(vertices, [], [])
            mesh.update()
            attr = mesh.attributes.new(name="fft", type='FLOAT', domain='POINT')
            for i in range(num_bands):
                attr.data[i].value = 0.0
        else:
            mesh = bpy.data.meshes.new(obj_name + "_Mesh")
            mesh.from_pydata([(0, 0, 0)], [], [])
            mesh.update()
            if mode == "VOLUME":
                mesh.attributes.new(name="volume", type='FLOAT', domain='POINT')
                mesh.attributes["volume"].data[0].value = 0.0
        obj = bpy.data.objects.new(obj_name, mesh)
        bpy.context.collection.objects.link(obj)
        if DEBUG:
            print(f"create_audio_source_object: Created object {obj_name}")
        return obj

def update_audio_source_attribute(mode, value):
    if mode == "VOLUME":
        obj_name = "Volume"
        if obj_name in bpy.data.objects:
            obj = bpy.data.objects[obj_name]
            try:
                value_rounded = round(value, 6)
                attr = obj.data.attributes["volume"]
                attr.data.foreach_set("value", [value_rounded])
                obj.data.update()
            except Exception as e:
                print("Error updating volume:", e)
    elif mode in ("NBANDS", "FFT_LOG"):
        obj_name = "FFT"
        if obj_name in bpy.data.objects:
            obj = bpy.data.objects[obj_name]
            try:
                value_rounded = [round(v, 6) for v in value]
                attr = obj.data.attributes["fft"]
                attr.data.foreach_set("value", value_rounded)
                obj.data.update()
            except Exception as e:
                print("Error updating FFT:", e)

def do_analysis(samples):
    props = bpy.context.scene.audio2bl_props
    gain = props.gain  
    gate = props.gate  
    mode = props.analysis_mode

    if mode == "VOLUME":
        # Calcola l'RMS del segnale
        rms = np.sqrt(np.dot(samples, samples) / samples.size)
        # Applica il gain direttamente all'RMS e moltiplica per AMPLIFICATION_FACTOR.
        # gain=0 mantiene il valore originale (fattore = 1) 
        amplified_rms = rms * (gain + 1) * AMPLIFICATION_FACTOR
        # Utilizza un range fisso per la normalizzazione: min=0 e max=1.
        norm_rms = max(0.0, min(amplified_rms, 1.0))
        if norm_rms < gate:
            return 0
        return norm_rms
    else:
        global fft_params
        spectrum_min = audio_config.get("spectrum_min", props.spectrum_min)
        spectrum_max = audio_config.get("spectrum_max", props.spectrum_max)
        num_bands = audio_config.get("num_bands", props.num_bands)
        if (fft_params is None or 
            len(samples) != len(fft_params["freqs"]) or 
            fft_params.get("spectrum_min") != spectrum_min or 
            fft_params.get("spectrum_max") != spectrum_max):
            fft_params = precalc_fft_params(len(samples), mode, num_bands, spectrum_min, spectrum_max)
        fft_vals = np.fft.rfft(samples)
        magnitudes = np.abs(fft_vals)
        raw_levels = []
        for i, idx in enumerate(fft_params["bands"]):
            band_value = np.mean(magnitudes[idx]) if np.any(idx) else 0
            lin_val = band_value ** 0.5
            raw_levels.append(lin_val)
        new_levels = [val * (gain + 1) for val in raw_levels]
        gated_levels = [0 if level < gate else min(level, 1) for level in new_levels]
        return gated_levels

def audio_callback(indata, frames, time_info, status):
    global latest_samples, last_sample_time
    if status:
        print("audio_callback: Stream status:", status)
    try:
        channel = audio_config.get("channel", 0)
        with data_lock:
            latest_samples = indata[:, channel].copy()
        last_sample_time = time.time()
        if DEBUG:
            print("audio_callback: Received samples, shape =", indata.shape)
    except Exception as e:
        print("audio_callback: Error extracting channel:", e)

def refresh_ui():
    global latest_samples, latest_levels, last_sample_time, audio_stream
    current_block = None
    with data_lock:
        if latest_samples is not None:
            current_block = latest_samples
            latest_samples = None
    if current_block is not None:
        new_levels = do_analysis(current_block)
        with data_lock:
            latest_levels = new_levels
    if latest_levels is not None:
        mode = bpy.context.scene.audio2bl_props.analysis_mode
        update_audio_source_attribute(mode, latest_levels)
        if isinstance(latest_levels, (tuple, list)):
            bpy.context.scene.audio2bl_props.input_level = round(np.mean(latest_levels), 6)
        else:
            bpy.context.scene.audio2bl_props.input_level = round(latest_levels, 6)
        if DEBUG:
            print("refresh_ui: Updated levels =", latest_levels)
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()
    
    if bpy.context.scene.audio2bl_props.sampling:
        if audio_stream is None or not getattr(audio_stream, "active", True):
            print("refresh_ui: Audio stream inactive, forced restart...")
            start_audio_stream(bpy.context.scene.audio2bl_props)
            last_sample_time = time.time()
        else:
            elapsed = time.time() - last_sample_time
            if elapsed > 5.0:
                print(f"refresh_ui: No audio signal for {elapsed:.1f} seconds. Forcing stream restart...")
                stop_audio_stream()
                start_audio_stream(bpy.context.scene.audio2bl_props)
                last_sample_time = time.time()
    
    fps = bpy.context.scene.audio2bl_props.refresh_fps
    if DEBUG:
        print("refresh_ui: Called. Next interval =", 1.0 / fps)
    return 1.0 / fps

def start_audio_stream(props):
    global audio_stream, audio_config, fft_params
    if audio_stream is not None:
        stop_audio_stream()
    # Salva la configurazione attuale (modalità, canale, range di frequenze e numero di bande se in modalità FFT)
    audio_config = {
        "analysis_mode": props.analysis_mode,
        "channel": props.channel,
    }
    if props.analysis_mode in ("NBANDS", "FFT_LOG"):
        audio_config["spectrum_min"] = props.spectrum_min
        audio_config["spectrum_max"] = props.spectrum_max
        audio_config["num_bands"] = props.num_bands
    fft_params = None
    create_audio_source_object(props.analysis_mode, props.num_bands if props.analysis_mode in ("NBANDS", "FFT_LOG") else None)
    try:
        device_index = int(props.device_list) if props.device_list != "none" else None
        device_info = sd.query_devices(device=device_index, kind='input')
        available_channels = int(device_info.get('max_input_channels', 1))
        if available_channels < 2:
            props.channel = 0
        audio_stream = sd.InputStream(
            device=device_index,
            channels=available_channels,
            samplerate=SAMPLERATE,
            blocksize=props.block_size,
            callback=audio_callback
        )
        audio_stream.start()
        print("start_audio_stream: Audio stream started.")
    except Exception as e:
        print("start_audio_stream: Error starting audio stream:", e)

def stop_audio_stream():
    global audio_stream
    if audio_stream is not None:
        try:
            audio_stream.stop()
            audio_stream.close()
            audio_stream = None
            print("stop_audio_stream: Audio stream stopped.")
        except Exception as e:
            print("stop_audio_stream: Error stopping audio stream:", e)

# ------------------------------------------------------------------------------
# Properties and settings (PropertyGroup)
# ------------------------------------------------------------------------------
class Audio2BlProperties(bpy.types.PropertyGroup):
    block_size: bpy.props.IntProperty(
        name="Block Size",
        description="Size of the audio block in frames",
        default=DEFAULT_BLOCK_SIZE,
        min=16,
        max=4096
    )
    refresh_fps: bpy.props.FloatProperty(
        name="FPS",
        description="Interface refresh rate (FPS)",
        default=DEFAULT_FPS,
        min=1.0,
        max=240.0
    )
    device_list: bpy.props.EnumProperty(
        name="Audio Device",
        description="Select the audio input device",
        items=lambda self, context: get_device_list(self, context) if sd is not None 
            else [("none", "sounddevice not installed", "Install sounddevice")],
    )
    input_level: bpy.props.FloatProperty(
        name="Input Level",
        description="Current audio level (0-1)",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR'
    )
    gain: bpy.props.FloatProperty(
        name="Gain",
        description="Additive gain applied to the signal",
        default=0.0,
        min=-10.0,
        max=10.0
    )
    gate: bpy.props.FloatProperty(
        name="Gate",
        description="Threshold to cut background noise",
        default=0.00,
        min=0.0,
        max=1.0,
        subtype='FACTOR'
    )
    # Le proprietà min_rms e max_rms sono state eliminate dal pannello.
    # Il range di normalizzazione è fisso a [0, 1].
    spectrum_min: bpy.props.FloatProperty(
        name="Min Frequency",
        description="Minimum frequency of the sampling range",
        default=20.0,
        min=0.0,
        max=SAMPLERATE/2
    )
    spectrum_max: bpy.props.FloatProperty(
        name="Max Frequency",
        description="Maximum frequency of the sampling range",
        default=20000.0,
        min=0.0,
        max=SAMPLERATE/2
    )
    analysis_mode: bpy.props.EnumProperty(
        name="Mode",
        description="Select the analysis mode",
        items=[
            ("VOLUME", "Volume", "Single volume analysis"),
            ("NBANDS", "FFT Linear", "FFT analysis with evenly spaced bands"),
            ("FFT_LOG", "FFT Log", "FFT analysis with logarithmically spaced bands"),
        ],
        default="VOLUME"
    )
    num_bands: bpy.props.IntProperty(
        name="Bands",
        description="Number of bands for FFT analysis",
        default=8,
        min=1,
        max=256
    )
    channel: bpy.props.IntProperty(
        name="Channel",
        description="Channel to sample (0 for first, 1 for second)",
        default=0,
        min=0,
        max=1,
    )
    sampling: bpy.props.BoolProperty(
        name="Sampling Active",
        description="Indicates if audio sampling is active",
        default=False
    )

def get_device_list(self, context):
    if sd is None:
        return [("none", "sounddevice not installed", "Install sounddevice")]
    try:
        devices = sd.query_devices()
    except Exception as e:
        print("get_device_list: Error querying devices:", e)
        return [("none", "Error querying devices", str(e))]
    items = []
    for i, dev in enumerate(devices):
        if dev.get('max_input_channels', 0) > 0:
            name = dev.get('name', f"Device {i}")
            items.append((str(i), name, ""))
    if not items:
        items = [("none", "No input device", "")]
    return items

# ------------------------------------------------------------------------------
# Load handler
# ------------------------------------------------------------------------------
import bpy.app.handlers

@bpy.app.handlers.persistent
def load_post_handler(dummy):
    if not bpy.app.timers.is_registered(refresh_ui):
        bpy.app.timers.register(refresh_ui, first_interval=(1.0 / DEFAULT_FPS))
        if DEBUG:
            print("load_post_handler: Timer re-registered after file load.")

# ------------------------------------------------------------------------------
# Sidebar Panel
# ------------------------------------------------------------------------------
class AUDIO2BL_PT_panel(bpy.types.Panel):
    bl_label = "Audio2Bl"
    bl_idname = "AUDIO2BL_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Audio2Bl"
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.audio2bl_props
        layout.prop(props, "block_size")
        layout.prop(props, "refresh_fps")
        layout.prop(props, "device_list")
        try:
            if props.device_list != "none":
                device_index = int(props.device_list)
                device_info = sd.query_devices(device=device_index, kind='input')
                if device_info.get('max_input_channels', 0) > 1:
                    layout.prop(props, "channel")
        except Exception as e:
            layout.label(text="Error retrieving device info")
        layout.prop(props, "gain")
        layout.prop(props, "gate")
        layout.prop(props, "analysis_mode", text="Mode")
        if props.analysis_mode in ("NBANDS", "FFT_LOG"):
            row = layout.row(align=True)
            row.label(text="Range")
            row.prop(props, "spectrum_min", text="Min")
            row.prop(props, "spectrum_max", text="Max")
            layout.prop(props, "num_bands", text="Bands")
        layout.separator()
        if props.sampling:
            layout.operator("audio2bl.toggle", text="Stop Audio Sampling")
        else:
            layout.operator("audio2bl.toggle", text="Start Audio Sampling")
        layout.separator()
        layout.label(text=f"Input Level: {props.input_level:.2f}")
        
class AUDIO2BL_OT_toggle(bpy.types.Operator):
    bl_idname = "audio2bl.toggle"
    bl_label = "Toggle Audio Sampling"
    
    def execute(self, context):
        props = context.scene.audio2bl_props
        if props.sampling:
            stop_audio_stream()
            props.sampling = False
            self.report({'INFO'}, "Audio sampling stopped.")
        else:
            start_audio_stream(props)
            props.sampling = True
            self.report({'INFO'}, "Audio sampling started.")
        return {'FINISHED'}

class AUDIO2BL_OT_start(bpy.types.Operator):
    bl_idname = "audio2bl.start"
    bl_label = "Start Audio2Bl"
    
    def execute(self, context):
        props = context.scene.audio2bl_props
        start_audio_stream(props)
        props.sampling = True
        self.report({'INFO'}, "Audio sampling started (see meter in panel and console).")
        return {'FINISHED'}

classes = (
    Audio2BlProperties,
    AUDIO2BL_PT_panel,
    AUDIO2BL_OT_toggle,
    AUDIO2BL_OT_start,
)

def register():
    global audio_stream, audio_config, latest_samples, latest_levels, fft_params
    audio_stream = None
    audio_config = {}
    latest_samples = None
    latest_levels = None
    fft_params = None

    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.audio2bl_props = bpy.props.PointerProperty(type=Audio2BlProperties)
    
    bpy.app.timers.register(refresh_ui, first_interval=(1.0 / DEFAULT_FPS))
    if load_post_handler not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(load_post_handler)
    
    print("Audio2Bl addon registered.")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, "audio2bl_props"):
        del bpy.types.Scene.audio2bl_props
    try:
        bpy.app.timers.unregister(refresh_ui)
    except Exception:
        pass
    if load_post_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(load_post_handler)
    print("Audio2Bl addon unregistered.")

if __name__ == "__main__":
    register()
