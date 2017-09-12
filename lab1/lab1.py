import cairo
import numpy
 
# A simple function to display an image in an ipython notebook
def nbimage( data ):
    from IPython.display import display, Image
    from PIL.Image import fromarray
    from StringIO import StringIO
 
    s = StringIO()
    fromarray( data ).save( s, 'png' )
    display( Image( s.getvalue() ) )
 
WIDTH = 512
HEIGHT = 288
 
# this is a numpy buffer to hold the image data
data = numpy.zeros( (HEIGHT,WIDTH,4), dtype=numpy.uint8 )
 
# this creates a cairo context based on the numpy buffer
ims = cairo.ImageSurface.create_for_data( data, cairo.FORMAT_ARGB32, WIDTH, HEIGHT )
cr = cairo.Context( ims )
 
# draw a blue line
cr.set_source_rgba( 1.0, 0.0, 0.0, 1.0 )
cr.set_line_width( 2.0 )
cr.move_to( 0.0, 0.0 )
cr.line_to( 100.0, 100.0 )
cr.stroke()
 
# display the image
nbimage( data )
