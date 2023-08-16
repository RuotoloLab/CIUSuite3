"""
Edited from PyGuBu SimpleToolTip.py at https://github.com/alejandroautalan/pygubu/blob/master/pygubu/widgets/simpletooltip.py
downloaded 9/19/2018. Original tooltip was causing problems with larger buttons used in CIUSuite2, so
this fork was developed to add more customization to the tooltip options.
PyGuBu copyright 2012-2016 Alejandro Autalan <alejandroautalan@gmail.com>
"""

# encoding: utf8
__all__ = ['ToolTip']


import tkinter as tk


class ToolTip(object):
    """
    Tooltip object to display some text above a widget
    """

    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self.text = None

    def showtip(self, text):
        """
        Display text in tooltip window.
        Updated to show the tooltip starting from the bottom left corner of the widget to
        prevent overlap with the widget, and a wraplength parameter added to the label to
        prevent long tooltips from continuing way offscreen.
        """
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx()
        y = y + cy + self.widget.winfo_rooty() + self.widget.winfo_height()
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        try:
            # For Mac OS
            tw.tk.call("::tk::unsupported::MacWindowStyle",
                       "style", tw._w,
                       "help", "noActivates")
        except tk.TclError:
            pass
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", foreground="black",
                         relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"),
                         wraplength=400)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def create(widget, text):
    tooltip = ToolTip(widget)

    def enter(event):
        tooltip.showtip(text)

    def leave(event):
        tooltip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


if __name__ == '__main__':
    root = tk.Tk()
    for idx in range(0, 2):
        b = tk.Button(root, text='A button')
        b.grid()
        create(b, 'A tooltip !!')
    root.mainloop()
