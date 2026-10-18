"""Editable vector schematic derived from the reported model implementations.

Canonical coordinates are points at final size. The same scene produces a live-
text SVG and a vector PDF, without reusing or editing a raster illustration.
"""
from pathlib import Path
import math
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT=Path(__file__).resolve().parent
W,H=480,325
font_dir=Path('C:/Windows/Fonts')
if (font_dir/'arial.ttf').exists():
    pdfmetrics.registerFont(TTFont('SchemaSans',str(font_dir/'arial.ttf')))
    pdfmetrics.registerFont(TTFont('SchemaSans-Bold',str(font_dir/'arialbd.ttf')))
    FONT,BOLD='SchemaSans','SchemaSans-Bold'
else:
    FONT,BOLD='Helvetica','Helvetica-Bold'
INK='#253444';LINE='#596C7E';GREY='#F1F4F7';BLUE='#1769A5';PALE='#EAF4FB'
scene=[]
def text(x,y,s,size=8,bold=False,anchor='middle',color=INK):
    scene.append(('text',x,y,s,size,bold,anchor,color))
def box(identity,x,y,w,h,lines,accent=False):
    scene.append(('box',identity,x,y,w,h,PALE if accent else GREY,BLUE if accent else LINE))
    step=11;start=y+h/2-(len(lines)-1)*step/2+2.7
    for i,label in enumerate(lines):
        assert stringWidth(label,BOLD if i==0 else FONT,8)<=w-10,(identity,label)
        text(x+w/2,start+i*step,label,8,bold=(i==0),color=BLUE if accent and i==0 else INK)
def arrow(identity,points,color=LINE):scene.append(('arrow',identity,points,color))
def circle(identity,x,y,r):scene.append(('circle',identity,x,y,r))

# (a) Main model: the quantum branch acts on all features, not on feature tokens.
text(8,13,'a  Main residual hybrid',10,True,'start')
text(472,13,'44,238 parameters',8,False,'end')
box('a-input',8,87,65,50,['359 features','atan inputs'])
box('a-backbone',95,43,95,42,['Window MLP','120 features'])
box('a-encoding',95,126,95,55,['Input encoding','512 amplitudes','9 grouped angles'])
box('a-readout',210,126,110,55,['9-qubit circuit','12 re-uploading layers','27 XYZ observables'],True)
box('a-concat',210,43,100,42,['Normalise + join','120 + 27 features'])
box('a-residual',330,43,95,42,['Residual MLP','147 to 128 to 10'])
circle('a-sum',450,98,11);text(450,101,'+',12)
box('a-output',397,137,76,42,['Prior correction','10-class output'])
arrow('a-input-backbone',[(73,107),(83,107),(83,64),(95,64)])
arrow('a-input-encoding',[(83,107),(83,153),(95,153)])
arrow('a-hidden',[(190,64),(210,64)])
arrow('a-quantum-input',[(190,153),(210,153)],BLUE)
arrow('a-readout-to-join',[(320,153),(325,153),(325,101),(260,101),(260,85)],BLUE)
arrow('a-residual-input',[(310,64),(330,64)])
arrow('a-residual-to-sum',[(425,64),(435,64),(435,98),(439,98)])
arrow('a-base-logits',[(142.5,43),(142.5,28),(450,28),(450,87)])
text(307,25,'Backbone classifier: 120 to 10 logits',7.5)
arrow('a-predict',[(450,109),(450,126),(435,126),(435,137)])
text(95,194,'Quantum branch: 324 rotation parameters + 216 re-uploading parameters',7.5,False,'start')
scene.append(('rule',8,204,472,204))

# (b) Older fixed fusion: independent classifiers, no shared residual head.
text(8,219,'b  Fixed score fusion',10,True,'start')
text(472,219,'55,581 reported scalars',8,False,'end')
box('b-input',8,252,65,35,['359 features'])
box('b-quantum',95,232,118,31,['Token-wise quantum','32,410 parameters'],True)
box('b-classical',95,275,118,31,['Window MLP','23,170 parameters'])
box('b-mean',250,248,106,42,['Prior-correct logits','0.5 / 0.5 mean'])
box('b-output',382,248,91,42,['10-class prediction'])
arrow('b-input-q',[(73,269),(83,269),(83,247.5),(95,247.5)])
arrow('b-input-c',[(83,269),(83,290.5),(95,290.5)])
arrow('b-q-mean',[(213,247.5),(230,247.5),(230,259),(250,259)],BLUE)
arrow('b-c-mean',[(213,290.5),(230,290.5),(230,280),(250,280)])
arrow('b-mean-output',[(356,269),(382,269)])
text(95,316,'Classical fusion control: Window MLP + Dense MLP; 0.6 / 0.4 probability mixture',7.5,False,'start')

svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}pt" height="{H}pt" viewBox="0 0 {W} {H}">',
     '<title>Reported residual hybrid and fixed score fusion</title>',
     '<desc>The main model uses global amplitude encoding with grouped angle re-uploading. The older model combines two independently trained classifiers by fixed score fusion.</desc>',
     f'<rect x="0" y="0" width="{W}" height="{H}" fill="white"/>']
pdf=canvas.Canvas(str(OUT/'fig_architecture.pdf'),pagesize=(W,H))
pdf.setTitle('HyQ-LC: residual hybrid and fixed score fusion')
def rgb(hexcolor):return tuple(int(hexcolor[i:i+2],16)/255 for i in (1,3,5))
def pdfline(points,color,width=1):
    pdf.setStrokeColorRGB(*rgb(color));pdf.setLineWidth(width)
    p=pdf.beginPath();p.moveTo(points[0][0],H-points[0][1])
    for x,y in points[1:]:p.lineTo(x,H-y)
    pdf.drawPath(p)
for item in scene:
    kind=item[0]
    if kind=='text':
        _,x,y,s,size,bold,anchor,color=item
        font=BOLD if bold else FONT;pdf.setFont(font,size);pdf.setFillColorRGB(*rgb(color))
        method={'middle':pdf.drawCentredString,'start':pdf.drawString,'end':pdf.drawRightString}[anchor]
        method(x,H-y,s)
        svg.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}pt" font-weight="{"bold" if bold else "normal"}" text-anchor="{anchor}" fill="{color}">{escape(s)}</text>')
    elif kind=='box':
        _,identity,x,y,w,h,fill,stroke=item
        pdf.setFillColorRGB(*rgb(fill));pdf.setStrokeColorRGB(*rgb(stroke));pdf.setLineWidth(.8)
        pdf.roundRect(x,H-y-h,w,h,4,fill=1,stroke=1)
        svg.append(f'<g id="{identity}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="0.8"/></g>')
    elif kind=='arrow':
        _,identity,points,color=item;pdfline(points,color)
        x,y=points[-1];x0,y0=points[-2];angle=math.atan2(y-y0,x-x0)
        head=[(x,y),(x-4*math.cos(angle-.45),y-4*math.sin(angle-.45)),(x-4*math.cos(angle+.45),y-4*math.sin(angle+.45))]
        pdf.setFillColorRGB(*rgb(color));p=pdf.beginPath();p.moveTo(head[0][0],H-head[0][1])
        for a,b in head[1:]:p.lineTo(a,H-b)
        p.close();pdf.drawPath(p,stroke=0,fill=1)
        coords=' '.join(f'{a},{b}' for a,b in points);tri=' '.join(f'{a},{b}' for a,b in head)
        svg.append(f'<g id="{identity}"><polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1"/><polygon points="{tri}" fill="{color}"/></g>')
    elif kind=='circle':
        _,identity,x,y,r=item;pdf.setFillColorRGB(1,1,1);pdf.setStrokeColorRGB(*rgb(LINE));pdf.circle(x,H-y,r,fill=1,stroke=1)
        svg.append(f'<circle id="{identity}" cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{LINE}"/>')
    elif kind=='rule':
        _,x,y,x2,y2=item;pdfline([(x,y),(x2,y2)],'#D7DFE5',.6)
        svg.append(f'<line x1="{x}" y1="{y}" x2="{x2}" y2="{y2}" stroke="#D7DFE5" stroke-width="0.6"/>')
svg.append('</svg>');(OUT/'fig_architecture.svg').write_text('\n'.join(svg)+'\n',encoding='utf-8')
pdf.save()
print('Wrote editable SVG and vector PDF with live text.')
