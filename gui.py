import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext as st
import os,sys,random,torch
import numpy as np
import threading
from diffusers import OnnxStableDiffusionPipeline

main_window = tk.Tk()
main_window.title("WaifuDiffusion AMD GUI - Tanuki")
main_window.geometry('720x480')
#main_window.resizable(False, False) 

# Diffusion Var
prompt = "night sky, blue hair, nekomimi"
negative_prompt = ""
denoiseStrength = 0.8 # a float number from 0 to 1 - decreasing this number will increase result similarity with baseImage
steps = 25
scale = 7.5
jumlah = 1 # how many you generate
folder = "jajal"
seed = 69
width = 512
height = 320

pipe = OnnxStableDiffusionPipeline.from_pretrained(r"F:\StableDiffusion\onnx",torch_dtype=torch.float16,revision="fp16",provider="DmlExecutionProvider")
pipe.safety_checker = None # Disable/Enable NSFW filter

last_out_image = "" # just to save a reference for the image, if not it will be destroyed by garbage collector, will result white image in tkinter PhotoImage

def set_log(text):
    log_text.config(state = "normal")
    log_text.insert('end', text)
    log_text.config(state = "disabled")
    log_text.see("end")

def set_diff_config():
    set_log("Config updated!")
    global prompt, negative_prompt, steps, scale, jumlah, folder, width, height, seed
    prompt = input_prompt.get()
    negative_prompt = input_nprompt.get()
    steps = int(input_steps.get())
    scale = float(input_scale.get())
    jumlah = int(input_many.get())
    folder = input_output.get()
    width = int(input_width.get())
    height = int(input_height.get())
    seed = int(input_seed.get())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_log(f"Generating {jumlah} Images\nWith Size : {width}x{height}\nand seed : {seed} inside : {folder} folder")

    if not os.path.exists(folder):
        os.makedirs(folder)
        set_log(f'No folder named {folder}.\nSuccessfully Created one.\n')

def generate():
    global last_out_image
    for x in range(int(jumlah)):
        image = pipe(prompt, width=width, height=height, strength=denoiseStrength, num_inference_steps=steps, guidance_scale=scale, negative_prompt=negative_prompt)
        image.images[0].save(f'{folder}\out_{str(x)}.png')
        last_out_image = tk.PhotoImage(file=f'{folder}\out_{str(x)}.png')
        img_label.configure(image=last_out_image)
        set_log(f'Image out_{str(x)}.png hasben generated!\n')
    set_log(f'Task Done..\n')
    submit["state"] = tk.NORMAL

def start_generate():
    set_log(f'Task Start...\n')
    submit["state"] = tk.DISABLED
    set_diff_config()
    threading.Thread(target=generate).start()












# Theme Config
background = "#000000"
foreground = "#FFFFFF"
logground = "#666666"
main_window.configure(bg=background)
ttk.Style().configure('Dark.Horizontal.TProgressbar',foreground=foreground,background=background)

# FRAME 1
frame01 = tk.Frame(main_window,borderwidth=5,bg=background)
label_prompt = tk.Label(frame01, text = "Prompt :",bg=background,fg=foreground)
input_prompt = tk.Entry(frame01, width = 99,bg=background,fg=foreground,insertbackground=foreground)
label_nprompt = tk.Label(frame01, text = "Negative Prompt :",bg=background,fg=foreground)
input_nprompt = tk.Entry(frame01, width = 99,bg=background,fg=foreground,insertbackground=foreground)
label_many = tk.Label(frame01, text = "How many Image? :",bg=background,fg=foreground)
input_many = tk.Entry(frame01, width = 5,bg=background,fg=foreground,insertbackground=foreground)
label_seed = tk.Label(frame01, text = "Seed :",bg=background,fg=foreground)
input_seed = tk.Entry(frame01, width = 10,bg=background,fg=foreground,insertbackground=foreground)
label_steps = tk.Label(frame01, text = "Steps :",bg=background,fg=foreground)
input_steps = tk.Entry(frame01, width = 5,bg=background,fg=foreground,insertbackground=foreground)
label_width = tk.Label(frame01, text = "Width :",bg=background,fg=foreground)
input_width = tk.Entry(frame01, width = 5,bg=background,fg=foreground,insertbackground=foreground)
label_height = tk.Label(frame01, text = "Height :",bg=background,fg=foreground)
input_height = tk.Entry(frame01, width = 5,bg=background,fg=foreground,insertbackground=foreground)
label_scale = tk.Label(frame01, text = "Scale :",bg=background,fg=foreground)
input_scale = tk.Entry(frame01, width = 5,bg=background,fg=foreground,insertbackground=foreground)
label_output = tk.Label(frame01, text = "Output Folder :",bg=background,fg=foreground)
input_output = tk.Entry(frame01, width = 10,bg=background,fg=foreground,insertbackground=foreground)
# FRAME 2
frame02 = tk.Frame(main_window,borderwidth=5,bg=background)
progress = ttk.Progressbar(frame02, orient='horizontal', mode='determinate', length=650, style='Dark.Horizontal.TProgressbar')
submit = tk.Button(frame02, text = "Generate!", command=start_generate,bg=background,fg=foreground)
# FRAME 3
frame03 = tk.Frame(main_window,borderwidth=5,bg=background)
img_label = tk.Label(frame03,text = "Last Generated Image Output")

# FRAME 4
frame04 = tk.Frame(main_window, borderwidth=5,bg=background)
log_text = st.ScrolledText(frame04, width=35, height=24,bg=logground, fg=foreground,state=tk.DISABLED)

frame01.pack(side=tk.TOP,fill=tk.BOTH,anchor=tk.NW)
label_prompt.grid(row=1,column=0,sticky=tk.W)
input_prompt.grid(row=1,column=1,sticky=tk.W,columnspan=15)
label_nprompt.grid(row=2,column=0,sticky=tk.W)
input_nprompt.grid(row=2,column=1,sticky=tk.W,columnspan=15)
label_many.grid(row=0,column=0,sticky=tk.W)
input_many.grid(row=0,column=1,sticky=tk.W)
label_steps.grid(row=0,column=2,sticky=tk.W)
input_steps.grid(row=0,column=3,sticky=tk.W)
label_scale.grid(row=0,column=4,sticky=tk.W)
input_scale.grid(row=0,column=5,sticky=tk.W)
label_width.grid(row=0,column=6,sticky=tk.W)
input_width.grid(row=0,column=7,sticky=tk.W)
label_height.grid(row=0,column=8,sticky=tk.W)
input_height.grid(row=0,column=9,sticky=tk.W)
label_seed.grid(row=0,column=10,sticky=tk.W)
input_seed.grid(row=0,column=11,sticky=tk.W)
label_output.grid(row=0,column=12,sticky=tk.W)
input_output.grid(row=0,column=13,sticky=tk.W)

frame02.pack(side=tk.BOTTOM,fill=tk.BOTH,anchor=tk.NW)
progress.grid(row=0,column=0,sticky=tk.W)
submit.grid(row=0,column=1,sticky=tk.W)

frame03.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
img_label.pack(fill=tk.BOTH,anchor=tk.CENTER)

frame04.pack(side=tk.RIGHT,fill=tk.BOTH,anchor=tk.NW,expand=False)
log_text.grid(row=0,column=1,sticky=tk.W)



# Set default input value
input_prompt.insert(0,prompt)
input_many.insert(0,jumlah)
input_seed.insert(0,seed)
input_steps.insert(0,steps)
input_scale.insert(0,scale)
input_width.insert(0,width)
input_height.insert(0,height)
input_output.insert(0,folder)



main_window.mainloop()
