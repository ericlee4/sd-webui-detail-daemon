import os
import random

import gradio as gr
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import modules.scripts as scripts
from modules.script_callbacks import on_cfg_denoiser, remove_callbacks_for_function, on_infotext_pasted, on_ui_settings
from modules.ui_components import InputAccordion
from modules import shared

try:
    import modules_forge.forge_version
    is_forge = True
except:
    is_forge = False

def add_settings():
    section = ('detail_daemon', "Detail Daemon")
    shared.opts.add_option("detail_daemon_count", shared.OptionInfo(
        6, "Daemon count", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section).needs_reload_ui())
    shared.opts.add_option("detail_daemon_verbose", shared.OptionInfo(
        False, "Verbose", gr.Checkbox, {"interactive": True}, section=section))

on_ui_settings(add_settings)

def parse_infotext(infotext, params):
    try:
        raw = params['Detail Daemon']
        if raw.startswith("D"):
            daemons = raw.split(";")
            if len(daemons) > shared.opts.data.get("detail_daemon_count", 6):
                tqdm.write(f"\033[31mDetail Daemon:\033[0m Need more daemons! Go to 'Settings > Uncategorized > Detail Daemon' and increase count to {len(daemons)}.")
            dd_dict = {}
            for idx, daemon in enumerate(daemons):
                tag, values = daemon.split(":")
                vals = values.split(",")
                dd_dict[f"active{idx + 1 if idx > 0 else ''}"] = bool(int(vals[0]))
                dd_dict[f"hr{idx + 1 if idx > 0 else ''}"] = bool(int(vals[1]))
                dd_dict[f"mode{idx + 1 if idx > 0 else ''}"] = vals[2]
                dd_dict[f"amount{idx + 1 if idx > 0 else ''}"] = float(vals[3])
                dd_dict[f"st{idx + 1 if idx > 0 else ''}"] = float(vals[4])
                dd_dict[f"ed{idx + 1 if idx > 0 else ''}"] = float(vals[5])
                dd_dict[f"bias{idx + 1 if idx > 0 else ''}"] = float(vals[6])
                dd_dict[f"exp{idx + 1 if idx > 0 else ''}"] = float(vals[7])
                dd_dict[f"st_offset{idx + 1 if idx > 0 else ''}"] = float(vals[8])
                dd_dict[f"ed_offset{idx + 1 if idx > 0 else ''}"] = float(vals[9])
                dd_dict[f"fade{idx + 1 if idx > 0 else ''}"] = float(vals[10])
                dd_dict[f"smooth{idx + 1 if idx > 0 else ''}"] = bool(int(vals[11]))
                dd_dict[f"noisetarget{idx + 1 if idx > 0 else ''}"] = vals[12]
                dd_dict[f"textcond_percent{idx + 1 if idx > 0 else ''}"] = float(vals[13])
                dd_dict[f"noise_size{idx + 1 if idx > 0 else ''}"] = float(vals[14])
                dd_dict[f"noise_upscale{idx + 1 if idx > 0 else ''}"] = vals[15]
                dd_dict[f"luminosity_threshold{idx + 1 if idx > 0 else ''}"] = float(vals[16])
            params['Detail Daemon'] = dd_dict
        else:
            # fallback to old format for backward compatibility
            d = {}
            for s in raw.split(','):
                k, _, v = s.partition(':')
                d[k.strip()] = v.strip()
            params['Detail Daemon'] = d
    except Exception:
        pass

on_infotext_pasted(parse_infotext)


def generate_noise(batch, channels, h, w, specified_noise_size, mode, dtype, interpolate, seed):
    noise_size = specified_noise_size**0.5
    low_h = max(1, int(h - (h - 1) * noise_size))
    low_w = max(1, int(w - (w - 1) * noise_size))
    gen = torch.Generator()
    if seed != -1:
        gen.manual_seed(seed)
    else:
        gen.seed()

    low_res = torch.randn((batch, channels, low_h, low_w), device="cpu", dtype=dtype, generator=gen)
    if interpolate:
        blobs = F.interpolate(low_res, size=(h, w), mode=mode)
        blobs = (blobs - blobs.mean()) / (blobs.std() + 1e-6)
        return blobs

    return low_res


def visualize_noise(noise_size, noise_upscale, interpolate, seed):
    noise_tensor = generate_noise(1, 16, 128, 128, noise_size, noise_upscale, torch.float32, interpolate, seed)
    img_data = noise_tensor[0, 0].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_data, cmap='viridis')
    plt.close(fig)
    return fig


class Script(scripts.Script):

    def __init__(self):
        self.tab_param_count = 0

    def title(self):
        return "Detail Daemon"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):

        def extract_infotext(d: dict, key, old_key=None):
            if 'Detail Daemon' in d:
                return d['Detail Daemon'].get(key)
            return d.get(old_key)

        daemon_count = shared.opts.data.get("detail_daemon_count", 6)
        all_params = []
        
        with InputAccordion(False, label="Detail Daemon", elem_id=self.elem_id('detail-daemon')) as gr_enabled:
            all_params.append(gr_enabled)
            self.infotext_fields = [(gr_enabled, lambda d: 'Detail Daemon' in d or 'DD_enabled' in d)]
            thumbs = []
            with gr.Group():
                with gr.Row(elem_classes=['detail-daemon-thumb-group']):
                    for i in range(daemon_count):
                        _, empty = self.visualize(False, 0, 1, 0.5, 0, 0, 0, 0, 0, False, 'both', False)
                        gr_thumb = gr.Plot(value=empty, elem_classes=['detail-daemon-thumb'], show_label=False)
                        thumbs.append(gr_thumb)
            with gr.Group(elem_classes=['detail-daemon-tab-group']):
                for i in range(daemon_count):
                    with gr.Tab(f'{["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"][i]}', elem_classes=['detail-daemon-tab']):
                        params_set = []
                        with gr.Row():
                            gr_active = gr.Checkbox(label="Active", value=False, min_width=60, elem_classes=['detail-daemon-active'])
                            gr_hires = gr.Checkbox(label="Hires Pass", value=False, min_width=60, elem_classes=['detail-daemon-hires'])
                            gr_noisetarget = gr.Radio(choices=[("Sigma", "sigma"), ("Latent", "latent"), ("Text Conditioning", "textcond")], value="sigma", label="Noise Target")
                        with gr.Row():
                            with gr.Column(scale=2, elem_classes=['detail-daemon-params']):
                                gr_amount_slider = gr.Slider(minimum=-1.00, maximum=1.00, step=.01, value=0.10, label="Detail Amount")
                                with gr.Row():
                                    gr_start = gr.Slider(minimum=0.0, maximum=1.0, step=.01, value=0.2, label="Start", min_width=60)
                                    gr_end = gr.Slider(minimum=0.0, maximum=1.0, step=.01, value=0.8, label="End", min_width=60)
                                with gr.Row():
                                    gr_start_offset_slider = gr.Slider(minimum=-1.00, maximum=1.00, step=.01, value=0.00, label="Start Offset", min_width=60)
                                    gr_end_offset_slider = gr.Slider(minimum=-1.00, maximum=1.00, step=.01, value=0.00, label="End Offset", min_width=60)
                                with gr.Row():
                                    gr_bias = gr.Slider(minimum=0.0, maximum=1.0, step=.01, value=0.5, label="Bias", min_width=60)
                                    gr_exponent = gr.Slider(minimum=0.0, maximum=10.0, step=.05, value=1.0, label="Exponent", min_width=60)
                                gr_fade = gr.Slider(minimum=0.0, maximum=1.0, step=.05, value=0.0, label="Fade", min_width=60)
                                with gr.Group(visible=False) as gr_textcond_settings:
                                    gr_textcond_percent = gr.Slider(minimum=0.0, maximum=1.0, step=.05, value=0.35, label="Text Cond Noise %",  min_width=60)
                                with gr.Group(visible=False) as gr_latent_settings:
                                    gr_noise_size = gr.Slider(minimum=0, maximum=1, step=0.001, value=.8, label="Noise Size", info='The size of the noise "blobs" as a percentage of the image size (exponential)')
                                    gr_noise_upscale = gr.Radio(choices=["nearest", "bilinear", "bicubic"], value="bicubic", label="Noise Upscale")
                                    gr_noise_seed = gr.Number(-1, label="Latent Noise Seed", minimum=-1, precision=0)

                                gr_luminosity_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=.05, value=0, label="Noise Luminosity Threshold", min_width=60)
                            with gr.Column(scale=1, min_width=275):
                                preview, _ = self.visualize(False, 0.2, 0.8, 0.5, 0.1, 1, 0, 0, 0, True, 'both', False)
                                gr_vis = gr.Plot(value=preview, elem_classes=['detail-daemon-vis'], show_label=False)
                                gr_smooth = gr.Checkbox(label="Smooth", value=True, min_width=60, elem_classes=['detail-daemon-smooth'])
                                fixed_seed = random.randint(0, 2**32 - 1)

                                def visualize_noise_with_seed(size, upscale, seed):
                                    use_seed = seed
                                    if use_seed == -1:
                                        use_seed = fixed_seed

                                    return [gr.Plot(visualize_noise(size, upscale, True, use_seed)), gr.Plot(visualize_noise(size, upscale, False, use_seed))]

                                gr_noise_plot = gr.Plot(visible=False, label="Upscaled Noise")
                                gr_lowres_noise_plot = gr.Plot(visible=False, label="Noise")
                                def update_fixed_seed(seed):
                                    if seed == -1:
                                        nonlocal fixed_seed
                                        fixed_seed = random.randint(0, 2**32 - 1)

                                gr_noise_seed.change(update_fixed_seed, inputs=[gr_noise_seed])
                                gr.on(
                                    triggers=[gr_noisetarget.change, gr_noise_size.release, gr_noise_upscale.change, gr_noise_seed.change],
                                    fn=visualize_noise_with_seed,
                                    inputs=[gr_noise_size, gr_noise_upscale, gr_noise_seed],
                                    outputs=[gr_noise_plot, gr_lowres_noise_plot]
                                )

                        with gr.Accordion("More Knobs:", elem_classes=['detail-daemon-more-accordion'], open=False):
                            with gr.Row():
                                with gr.Column(scale=2):
                                    with gr.Row():
                                        # Because the slider max and min are sometimes too limiting:
                                        gr_amount = gr.Number(value=0.10, precision=4, step=.01, label="Amount", min_width=60)
                                        gr_start_offset = gr.Number(value=0.0, precision=4, step=.01, label="Start Offset", min_width=60)
                                        gr_end_offset = gr.Number(value=0.0, precision=4, step=.01, label="End Offset", min_width=60)
                                        gr_mode = gr.Dropdown(["both", "cond", "uncond"], value="both", label="Mode", show_label=True, min_width=60, elem_classes=['detail-daemon-mode'])

                        gr_amount_slider.release(None, gr_amount_slider, gr_amount, _js="(x) => x")
                        gr_amount.change(None, gr_amount, gr_amount_slider, _js="(x) => x")

                        gr_start_offset_slider.release(None, gr_start_offset_slider, gr_start_offset, _js="(x) => x")
                        gr_start_offset.change(None, gr_start_offset, gr_start_offset_slider, _js="(x) => x")

                        gr_end_offset_slider.release(None, gr_end_offset_slider, gr_end_offset, _js="(x) => x")
                        gr_end_offset.change(None, gr_end_offset, gr_end_offset_slider, _js="(x) => x")

                        vis_args = [gr_active, gr_start, gr_end, gr_bias, gr_amount, gr_exponent, gr_start_offset, gr_end_offset, gr_fade, gr_smooth, gr_mode, gr_hires]
                        for vis_arg in vis_args:
                            if isinstance(vis_arg, gr.components.Slider):
                                vis_arg.release(fn=self.visualize, show_progress=False, inputs=vis_args, outputs=[gr_vis, thumbs[i]])
                            else:
                                vis_arg.change(fn=self.visualize, show_progress=False, inputs=vis_args, outputs=[gr_vis, thumbs[i]])

                        def update_noise_target_visibility(target):
                            is_textcond = target == "textcond"
                            is_latent = target == "latent"
                            return [
                                gr.Group(visible=is_textcond),
                                gr.Group(visible=is_latent),
                                gr.Plot(visible=is_latent),
                                gr.Plot(visible=is_latent),
                            ]

                        gr_noisetarget.change(fn=update_noise_target_visibility, inputs=[gr_noisetarget], outputs=[gr_textcond_settings, gr_latent_settings, gr_noise_plot, gr_lowres_noise_plot])


                        params_set = [
                            gr_active, gr_hires, gr_mode, gr_start, gr_end, gr_bias, gr_amount,
                            gr_exponent, gr_start_offset, gr_end_offset, gr_fade, gr_smooth, gr_noisetarget,
                            gr_textcond_percent, gr_noise_size, gr_noise_upscale, gr_noise_seed, gr_luminosity_threshold
                        ]
                        all_params.extend(params_set)

                        # First tab should be backward compatible with the older single daemon Detail Daemon, hence the variable names without numbers
                        # older single daemon DD was backward compatible with yet older DD which had infotext with the DD_stuff format, so that's handled here too
                        if i == 0 :
                            self.tab_param_count = len(params_set)
                            self.infotext_fields.extend([
                                (gr_active, lambda d: extract_infotext(d, 'active') or 'Detail Daemon' in d or 'DD_enabled' in d),
                                (gr_hires, lambda d: extract_infotext(d, 'hr') or False),
                                (gr_mode, lambda d: extract_infotext(d, 'mode', 'DD_mode')),
                                (gr_amount, lambda d: extract_infotext(d, 'amount', 'DD_amount')),
                                (gr_start, lambda d: extract_infotext(d, 'st', 'DD_start')),
                                (gr_end, lambda d: extract_infotext(d, 'ed', 'DD_end')),
                                (gr_bias, lambda d: extract_infotext(d, 'bias', 'DD_bias')),
                                (gr_exponent, lambda d: extract_infotext(d, 'exp', 'DD_exponent')),
                                (gr_start_offset, lambda d: extract_infotext(d, 'st_offset', 'DD_start_offset')),
                                (gr_end_offset, lambda d: extract_infotext(d, 'ed_offset', 'DD_end_offset')),
                                (gr_fade, lambda d: extract_infotext(d, 'fade', 'DD_fade')),
                                (gr_smooth, lambda d: extract_infotext(d, 'smooth', 'DD_smooth')),
                                (gr_noisetarget, lambda d: extract_infotext(d, 'noisetarget')),
                                (gr_textcond_percent, lambda d: extract_infotext(d, 'textcond_percent')),
                                (gr_noise_size, lambda d: extract_infotext(d, 'noise_size') or 0.8),
                                (gr_noise_upscale, lambda d: extract_infotext(d, 'noise_upscale')),
                                (gr_noise_seed, lambda d: extract_infotext(d, 'noise_seed')),
                                (gr_luminosity_threshold, lambda d: extract_infotext(d, 'luminosity_threshold')),
                            ])
                        else:
                            tab_tag = i + 1
                            self.infotext_fields.extend([
                                (gr_active, lambda d, key=f'active{tab_tag}': extract_infotext(d, key) or False),
                                (gr_hires, lambda d, key=f'hr{tab_tag}': extract_infotext(d, key)),
                                (gr_mode, lambda d, key=f'mode{tab_tag}': extract_infotext(d, key)),
                                (gr_amount, lambda d, key=f'amount{tab_tag}': extract_infotext(d, key)),
                                (gr_start, lambda d, key=f'st{tab_tag}': extract_infotext(d, key)),
                                (gr_end, lambda d, key=f'ed{tab_tag}': extract_infotext(d, key)),
                                (gr_bias, lambda d, key=f'bias{tab_tag}': extract_infotext(d, key)),
                                (gr_exponent, lambda d, key=f'exp{tab_tag}': extract_infotext(d, key)),
                                (gr_start_offset, lambda d, key=f'st_offset{tab_tag}': extract_infotext(d, key)),
                                (gr_end_offset, lambda d, key=f'ed_offset{tab_tag}': extract_infotext(d, key)),
                                (gr_fade, lambda d, key=f'fade{tab_tag}': extract_infotext(d, key)),
                                (gr_smooth, lambda d, key=f'smooth{tab_tag}': extract_infotext(d, key)),
                                (gr_noisetarget, lambda d, key=f'noisetarget{tab_tag}': extract_infotext(d, key)),
                                (gr_textcond_percent, lambda d, key=f'textcond_percent{tab_tag}': extract_infotext(d, key)),
                                (gr_noise_size, lambda d, key=f'noise_size{tab_tag}': extract_infotext(d, key)),
                                (gr_noise_upscale, lambda d, key=f'noise_upscale{tab_tag}': extract_infotext(d, key)),
                                (gr_noise_seed, lambda d, key=f'noise_seed{tab_tag}': extract_infotext(d, key)),
                                (gr_luminosity_threshold, lambda d, key=f'luminosity_threshold{tab_tag}': extract_infotext(d, key)),
                            ])
        return all_params

    def process(self, p, enabled, *all_daemon_args):
        if not enabled:
            if hasattr(self, 'callback_added'):
                remove_callbacks_for_function(self.denoiser_callback)
                delattr(self, 'callback_added')
            return

        if p.sampler_name in ["DPM adaptive", "HeunPP2"]:
            tqdm.write(f'\033[31mDetail Daemon:\033[0m Selected sampler ({p.sampler_name}) is not supported.')
            return

        self.daemon_data = []
        extra_gen_texts = []
        num_daemons = len(all_daemon_args) // self.tab_param_count

        for i in range(num_daemons):
            start_idx = i * self.tab_param_count
            end_idx = start_idx + self.tab_param_count
            daemon_args = all_daemon_args[start_idx:end_idx]

            active, hires, mode, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth, noisetarget, textcond_percent, noise_size, noise_upscale, noise_seed, luminosity_threshold = daemon_args

            # TODO? XYZ support for other channels
            if (i == 0) :
                mode = getattr(p, "DD_mode", mode)
                amount = getattr(p, "DD_amount", amount)
                start = getattr(p, "DD_start", start)
                end = getattr(p, "DD_end", end)
                bias = getattr(p, "DD_bias", bias)
                exponent = getattr(p, "DD_exponent", exponent)
                start_offset = getattr(p, "DD_start_offset", start_offset)
                end_offset = getattr(p, "DD_end_offset", end_offset)
                fade = getattr(p, "DD_fade", fade)
                smooth = getattr(p, "DD_smooth", smooth)
                noisetarget = getattr(p, "DD_noisetarget", noisetarget)
                textcond_percent = getattr(p, "DD_textcond_percent", textcond_percent)
                noise_size = getattr(p, "DD_noise_size", noise_size)
                noise_upscale = getattr(p, "DD_noise_upscale", noise_upscale)
                noise_seed = getattr(p, "DD_noise_seed", noise_seed)
                luminosity_threshold = getattr(p, "DD_luminosity_threshold", luminosity_threshold)

            if active:
                daemon_schedule_params = {
                    "start": start,
                    "end": end,
                    "bias": bias,
                    "amount": amount,
                    "exponent": exponent,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "fade": fade,
                    "smooth": smooth
                }

                self.daemon_data.append({
                    'name': f'Daemon {i+1}',
                    'mode': mode,
                    'schedule': None,
                    'schedule_params': daemon_schedule_params,
                    'hires': hires,
                    'multiplier': .1,  # Add slider for this?
                    'noisetarget': noisetarget,
                    'textcond_percent': textcond_percent,
                    'noise_size': noise_size,
                    'noise_upscale': noise_upscale,
                    'noise_seed': noise_seed,
                    'luminosity_threshold': luminosity_threshold
                })

                text = ",".join([
                    str(int(active)), str(int(hires)), mode, f"{amount}", f"{start}", f"{end}", f"{bias}",
                    f"{exponent}", f"{start_offset}", f"{end_offset}", f"{fade:}", str(int(smooth)),
                    noisetarget, str(textcond_percent), str(noise_size), noise_upscale, str(noise_seed), str(luminosity_threshold)
                ])
                extra_gen_texts.append(f"D{i+1}:{text}")

        if extra_gen_texts:
            p.extra_generation_params['Detail Daemon'] = ";".join(extra_gen_texts)

        if not hasattr(self, 'callback_added'):
            on_cfg_denoiser(self.denoiser_callback)
            self.callback_added = True
        self.cfg_scale = p.cfg_scale
        self.batch_size = p.batch_size
        self.is_hires_pass = False

    def before_process_batch(self, p, *args, **kwargs):
        self.is_hires_pass = False

    def before_hr(self, p, *args):
        self.is_hires_pass = True

    def postprocess(self, p, processed, *args):
        if hasattr(self, 'callback_added'):
            remove_callbacks_for_function(self.denoiser_callback)
            delattr(self, 'callback_added')

    def denoiser_callback(self, params):
        for daemon in self.daemon_data:
            if daemon['hires'] != self.is_hires_pass:
                continue

            name = daemon['name']
            mode = daemon['mode']
            step = max(params.sampling_step, params.denoiser.step)
            steps = max(params.total_sampling_steps, params.denoiser.total_steps)
            actual_steps = steps - max(steps // params.denoiser.steps - 1, 0)
            idx = min(step, actual_steps - 1)

            if daemon['schedule'] is None:
                daemon['schedule'] = self.make_schedule(actual_steps, **daemon['schedule_params'])

            schedule = daemon['schedule']
            multiplier = schedule[idx] * daemon['multiplier']
            noisetarget = daemon['noisetarget']

            if is_forge:
                if idx == 0 and mode != "both":
                    tqdm.write(f'\033[33mDetail Daemon:\033[0m Forge does not support `cond` and `uncond` modes, using `both` instead')
                mode = "both"

            if mode == "cond":
                for i in range(self.batch_size):
                    params.sigma[i] *= 1 - multiplier
            elif mode == "uncond":
                for i in range(self.batch_size):
                    params.sigma[self.batch_size + i] *= 1 + multiplier
            else:
                if noisetarget == "textcond":
                    cond: torch.Tensor = params.text_cond
                    b, t, c = cond.shape
                    rms = cond.detach().float().pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
                    noise = torch.rand((b, t, c), dtype=torch.float32, device=cond.device) * 2.0 - 1.0
                    noise = (noise * (schedule[idx] * rms)).to(cond.dtype)
                    noise_mask = torch.bernoulli(
                        torch.full((b, t, 1), daemon["textcond_percent"], dtype=torch.float32, device=cond.device)
                    ).to(cond.dtype)
                    params.text_cond = cond.detach() + noise * noise_mask
                elif noisetarget == "latent":
                    batch, channels, h, w = params.x.shape
                    noise_size = daemon["noise_size"]
                    noise_seed = daemon["noise_seed"]
                    blobs = generate_noise(batch, channels, h, w, noise_size, daemon["noise_upscale"], params.x.dtype, True, noise_seed if noise_seed != -1 else params.denoiser.p.seed)
                    blobs = blobs.to(params.x.device)
                    blobs *= params.x.std()
                    scheduleval = float(schedule[idx])
                    params.x = (1.0 - scheduleval**2)**0.5 * params.x + scheduleval * blobs
                else:
                    threshold = daemon.get("luminosity_threshold", 0)

                    if threshold == 0:
                        params.sigma *= 1 - multiplier * self.cfg_scale
                    else:
                        latent_brightness = params.x[:, 0, :, :].detach()
                        b_min, b_max = latent_brightness.min(), latent_brightness.max()
                        lum_mask = (latent_brightness - b_min) / (b_max - b_min + 1e-6)
                        steepness = 10.0
                        soft_mask = torch.sigmoid(steepness * (lum_mask - threshold))
                        num_channels = params.x.shape[1]
                        final_mask = soft_mask.unsqueeze(1).repeat(1, num_channels, 1, 1)
                        params.x = params.x * (1 - (schedule[idx] * final_mask))

            if shared.opts.data.get("detail_daemon_verbose", False):
                tqdm.write(f'\033[32mDetail Daemon:\033[0m {name} | sigma: {params.sigma} | step: {idx}/{actual_steps - 1} | multiplier: {multiplier:.4f}')


    def make_schedule(self, steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth):
        start = min(start, end)
        mid = start + bias * (end - start)
        multipliers = np.zeros(steps)

        start_idx, mid_idx, end_idx = [int(round(x * (steps - 1))) for x in [start, mid, end]]

        start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
        if smooth:
            start_values = 0.5 * (1 - np.cos(start_values * np.pi))
        start_values = start_values ** exponent
        if start_values.any():
            start_values *= (amount - start_offset)
            start_values += start_offset

        end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
        if smooth:
            end_values = 0.5 * (1 - np.cos(end_values * np.pi))
        end_values = end_values ** exponent
        if end_values.any():
            end_values *= (amount - end_offset)
            end_values += end_offset

        multipliers[start_idx:mid_idx+1] = start_values
        multipliers[mid_idx:end_idx+1] = end_values
        multipliers[:start_idx] = start_offset
        multipliers[end_idx+1:] = end_offset

        mask = multipliers > 0
        multipliers[mask] *= np.linspace(1.0, 1.0 - fade, mask.sum())

        return multipliers

    def visualize(self, enabled, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth, mode, hires):
        try:
            steps = 50
            values = self.make_schedule(steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth)
            mean = sum(values)/steps
            peak = np.clip(max(abs(values)), -1, 1)
            if start > end:
                start = end
            mid = start + bias * (end - start)
            plot_color = (0.5, 0.5, 0.5, 1) if not enabled else ((1 - peak)**2, 1, 0, 1) if mean >= 0 else (1, (1 - peak)**2, 0, 1)

            plt.rcParams.update({
                "text.color":  plot_color,
                "axes.labelcolor":  plot_color,
                "axes.edgecolor":  plot_color,
                "figure.facecolor":  (0.0, 0.0, 0.0, 0.0),
                "axes.facecolor":    (0.0, 0.0, 0.0, 0.0),
                "ytick.labelsize": 6,
                "ytick.labelcolor": plot_color,
                "ytick.color": plot_color,
            })

            fig_main, ax_main = plt.subplots(figsize=(2.15, 2.00), layout="constrained")
            ax_main.plot(range(steps), values, color=plot_color, linewidth=1.5, linestyle="dashed" if hires else "solid")
            ax_main.axhline(y=0, color=plot_color, linestyle='dotted')
            ax_main.axvline(x=mid * (steps - 1), color=plot_color, linestyle='dotted')
            ax_main.tick_params(right=False, color=plot_color)
            ax_main.set_xticks([i * (steps - 1) / 10 for i in range(10)][1:])
            ax_main.set_xticklabels([])
            ax_main.set_ylim([-1, 1])
            ax_main.set_xlim([0, steps - 1])
            plt.close(fig_main)

            plot_color = (0.5, 0.5, 0.5, .1) if not enabled else (0.75, 0.75, 0.75, 1)
            plt.rcParams.update({
                "text.color":  plot_color,
                "axes.labelcolor":  plot_color,
                "axes.edgecolor":  plot_color,
            })

            fig_thumb, ax_thumb = plt.subplots(figsize=(0.85, 0.85), layout="constrained")
            ax_thumb.plot(range(steps), values, color=plot_color, linewidth=1.5, linestyle="dashed" if hires else "solid")
            ax_thumb.set_xticks([])
            ax_thumb.set_yticks([])
            ax_thumb.set_ylim([-1, 1])
            ax_thumb.set_xlim([0, steps - 1])
            if (mode != "both"):
                ax_thumb.text(
                    0.98, 0.96, mode.upper(),
                    transform=ax_thumb.transAxes,
                    fontsize=8, fontweight='bold', color=plot_color,
                    ha='right', va='top'
                )
            plt.close(fig_thumb)

            self.last_vis = fig_main
            self.last_thumb = fig_thumb
            return [fig_main, fig_thumb]
        except Exception:
            if self.last_vis is not None and self.last_thumb is not None:
                return [self.last_vis, self.last_thumb]
            return

def xyz_support():
    for scriptDataTuple in scripts.scripts_data:
        if os.path.basename(scriptDataTuple.path) == 'xyz_grid.py':
            xy_grid = scriptDataTuple.module

            def confirm_mode(p, xs):
                for x in xs:
                    if x not in ['both', 'cond', 'uncond']:
                        raise RuntimeError(f'Invalid Detail Daemon Mode: {x}')
            mode = xy_grid.AxisOption(
                '[Detail Daemon] Mode',
                str,
                xy_grid.apply_field('DD_mode'),
                confirm=confirm_mode
            )
            amount = xy_grid.AxisOption(
                '[Detail Daemon] Amount',
                float,
                xy_grid.apply_field('DD_amount')
            )
            start = xy_grid.AxisOption(
                '[Detail Daemon] Start',
                float,
                xy_grid.apply_field('DD_start')
            )
            end = xy_grid.AxisOption(
                '[Detail Daemon] End',
                float,
                xy_grid.apply_field('DD_end')
            )
            bias = xy_grid.AxisOption(
                '[Detail Daemon] Bias',
                float,
                xy_grid.apply_field('DD_bias')
            )
            exponent = xy_grid.AxisOption(
                '[Detail Daemon] Exponent',
                float,
                xy_grid.apply_field('DD_exponent')
            )
            start_offset = xy_grid.AxisOption(
                '[Detail Daemon] Start Offset',
                float,
                xy_grid.apply_field('DD_start_offset')
            )
            end_offset = xy_grid.AxisOption(
                '[Detail Daemon] End Offset',
                float,
                xy_grid.apply_field('DD_end_offset')
            )
            fade = xy_grid.AxisOption(
                '[Detail Daemon] Fade',
                float,
                xy_grid.apply_field('DD_fade')
            )
            smooth = xy_grid.AxisOption(
                '[Detail Daemon] Smooth',
                bool,
                xy_grid.apply_field('DD_smooth')
            )
            xy_grid.axis_options.extend([
                mode,
                amount,
                start,
                end,
                bias,
                exponent,
                start_offset,
                end_offset,
                fade,
                smooth,
            ])


try:
    xyz_support()
except Exception as e:
    tqdm.write(f'f"\033[31mDetail Daemon:\033[0m Error trying to add XYZ plot options for Detail Daemon: {e}')
