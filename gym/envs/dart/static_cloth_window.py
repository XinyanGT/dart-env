# Contributors: Alexander Clegg (alexanderwclegg@gmail.com)

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT
import sys
import numpy as np
from pydart2.gui.opengl.scene import OpenGLScene
from pydart2.gui.glut.window import *

from pyPhysX.clothHandles import *

from gym.envs.dart.static_window import *

import pyPhysX.pyutils as pyutils

class StaticClothGLUTWindow(StaticGLUTWindow):
    def __init__(self, sim, title, clothScene=None, extraRenderFunc=None, inputFunc=None, resetFunc=None, env=None):
        super(StaticClothGLUTWindow,self).__init__(sim, title)
        self.clothScene = clothScene
        self.env = env
        self.extraRenderFunc = extraRenderFunc
        self.inputFunc = inputFunc
        self.resetFunc = resetFunc
        self.mouseLButton = False
        self.mouseRButton = False
        #store the most recent rendering matrices
        self.modelviewM = None
        self.projectionM = None
        self.viewport = None
        self.prevWorldMouse = np.array([0.,0,0])
        self.camForward = np.array([0.,0,1])
        self.camForwardNear = np.array([0.,0.,0.])
        self.camUp = np.array([0.,1.0,0])
        self.camRight = np.array([1.,0.0,0])
        self.mouseLastPos = np.array([0,0])
        #self.mouseCurPos = np.array([0,0])
        self.curInteractorIX = None #set an interactor class to change user interaction with the window
        self.interactors = []
        [BaseInteractor(self), VertexSelectInteractor(self), FrameSelectInteractor(self), IKInteractor(self), PoseInteractor(self), Frame6DInteractor(self)] #the list of available interactors
        #self.interactors.append(BaseInteractor(self))
        #self.interactors.append(VertexSelectInteractor(self))
        self.lastContextSwitch = 0 #holds the frame of the last context switch (for label rendering)
        self.captureIndex = 0 #increments when captureToFile is called
        self.captureDirectory = "/home/alexander/Documents/frame_capture_output/demos/01_onearm_standard_noactivecompliance_strconpen_MPCobs_damped"
        #self.captureDirectory = "/home/alexander/Documents/frame_capture_output/variations/1"
        #self.captureDirectory = "/home/alexander/Documents/frame_capture_output/variations/elbow_data/0"
        #self.captureDirectory = "/home/alexander/Documents/dev/saved_render_states/siggraph_asia_finals/tshirt_failures/frames"
        self.capturing = False
        self.key_down = []
        for i in range(256):
            self.key_down.append(False)

    def run(self, _width=None, _height=None, _show_window=True ):
        # Init glut
        self._show_window = _show_window
        GLUT.glutInit(())
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA |
                                 GLUT.GLUT_DOUBLE |
                                 GLUT.GLUT_ALPHA |
                                 GLUT.GLUT_DEPTH)
        if _width is not None and _height is not None:
            GLUT.glutInitWindowSize(_width, _height)
            # self.resizeGL(_width, _height) # this line crashes my program ??
        else:
            GLUT.glutInitWindowSize(*self.window_size)
        GLUT.glutInitWindowPosition(0, 0)
        self.window = GLUT.glutCreateWindow(self.title)
        if not _show_window:
            GLUT.glutHideWindow()

        self.viewport = GL.glGetInteger(GL.GL_VIEWPORT)
        self.interactors = [BaseInteractor(self), VertexSelectInteractor(self), FrameSelectInteractor(self), IKInteractor(self), PoseInteractor(self), Frame6DInteractor(self)]  # the list of available interactors

        GLUT.glutDisplayFunc(self.drawGL)
        GLUT.glutReshapeFunc(self.resizeGL)
        GLUT.glutKeyboardFunc(self.mykeyboard)
        GLUT.glutKeyboardUpFunc(self.keyboardUp)
        GLUT.glutMouseFunc(self.mouseFunc)
        GLUT.glutMotionFunc(self.motionFunc)
        GLUT.glutPassiveMotionFunc(self.passiveMotionFunc)
        self.initGL(*self.window_size)

    def captureToFile(self, directory):
        #print("capture! index = %d" % self.captureIndex)
        from PIL import Image
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        w = self.viewport[2]
        h = self.viewport[3]
        #w, h = 1280, 720
        data = GL.glReadPixels(0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        img = Image.frombytes("RGB", (w, h), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        filename = directory + "/capture%05d.png" % self.captureIndex
        img.save(filename, 'png')
        self.captureIndex += 1

    def resizeGL(self, w, h):
        self.scene.resize(w, h)
        self.interactors[4].defineBoxRanges()
        
    def extraRender(self):
        'Place any extra rendering functionality here. This can be used to extend the window class'
        #get the window camera transformations
        GL.glLoadIdentity()
        GL.glTranslate(*self.scene.tb.trans)
        GL.glMultMatrixf(self.scene.tb.matrix)
        #GLUT.glutSolidSphere(0.05,10,10) #testing origin location

        #grab the current rendering params
        self.modelviewM = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
        self.projectionM = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
        self.viewport = GL.glGetInteger(GL.GL_VIEWPORT)

        if self.extraRenderFunc is not None:
            self.extraRenderFunc()

        if self.clothScene is not None:
            self.clothScene.render()

        #unprojections:
        #mouse hover object
        z = GL.glReadPixels(self.mouseLastPos[0],self.viewport[3]-self.mouseLastPos[1], 1, 1, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        self.prevWorldMouse = np.array(self.unproject(np.array([self.mouseLastPos[0],self.viewport[3]-self.mouseLastPos[1], z])))

        #camera forward
        z = GL.glReadPixels(self.viewport[2]/2, self.viewport[3]/2, 1, 1, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        forwardNear = np.array(self.unproject(np.array([self.viewport[2]/2, self.viewport[3]/2, 0])))
        self.camForwardNear = forwardNear
        forwardFar = np.array(self.unproject(np.array([self.viewport[2]/2, self.viewport[3]/2, z])))
        self.camForward = forwardFar-forwardNear
        self.camForward = self.camForward / np.linalg.norm(self.camForward)
        #camera right
        rightNear = np.array(self.unproject(np.array([self.viewport[2] - 1, self.viewport[3] / 2, 0])))
        self.camRight = rightNear - forwardNear
        self.camRight = self.camRight / np.linalg.norm(self.camRight)
        #camera up
        upNear = np.array(self.unproject(np.array([self.viewport[2]/2, self.viewport[3]-1, 0])))
        self.camUp = upNear - forwardNear
        self.camUp = self.camUp / np.linalg.norm(self.camUp)

        #self.clothScene.drawText(x=self.viewport[2] / 2, y=self.viewport[3] - 60, text="Forward = " + str(self.camForward), color=(0., 0, 0))
        #self.clothScene.drawText(x=self.viewport[2] / 2, y=self.viewport[3] - 75, text="Up = " + str(self.camUp), color=(0., 0, 0))
        #self.clothScene.drawText(x=self.viewport[2] / 2, y=self.viewport[3] - 90, text="Right = " + str(self.camRight), color=(0., 0, 0))

        #self.drawLine(p0=self.camForward)
        #self.drawLine(p0=self.camUp)
        #self.drawLine(p0=self.camRight)


        if self.lastContextSwitch < 50:
            context = "Default Context"
            if self.curInteractorIX is not None:
                context = self.interactors[self.curInteractorIX].label
            self.clothScene.drawText(x=self.viewport[2]/2, y=self.viewport[3]-30, text="Active Context = " + str(context), color=(0., 0, 0))

        if self.curInteractorIX is not None:
            self.interactors[self.curInteractorIX].contextRender()

        if self.capturing:
            self.captureToFile(directory=self.captureDirectory)

        self.lastContextSwitch += 1

    def mouseFunc(self, button, state, x, y):
        if self.curInteractorIX is not None:
            self.interactors[self.curInteractorIX].click(button, state, x, y)
            return

        tb = self.scene.tb
        #print(button)
        if state == 0:  # Mouse pressed
            if button == 0:
                self.mouseLButton = True
            if button == 2:
                self.mouseRButton = True
            self.mouseLastPos = np.array([x, y])
            if button == 3:
                tb.trans[2] += 0.1
            elif button == 4:
                tb.trans[2] -= 0.1
        elif state == 1:
            #self.mouseLastPos = None
            if button == 0:
                self.mouseLButton = False
            if button == 2:
                self.mouseRButton = False

    def motionFunc(self, x, y):
        if self.curInteractorIX is not None:
            self.interactors[self.curInteractorIX].drag(x, y)
            return

        dx = x - self.mouseLastPos[0]
        dy = y - self.mouseLastPos[1]
        modifiers = GLUT.glutGetModifiers()
        tb = self.scene.tb
        #print(tb.trans)
        # print("mouse motion: " + str(dx))
        #print("SHIFT?" + str(modifiers))
        if(self.key_down[120]): #'x' is down
            try:
                self.env.sawyer_root_dofs[0] += dx*0.01
                print(self.env.sawyer_root_dofs[:3])
            except:
                a=0
        if(self.key_down[121]): #'y' is down
            try:
                self.env.sawyer_root_dofs[1] += dx*0.01
                print(self.env.sawyer_root_dofs[:3])
            except:
                a=0
        if(self.key_down[122]): #'z' is down
            try:
                self.env.sawyer_root_dofs[2] += dx*0.01
                print(self.env.sawyer_root_dofs[:3])
            except:
                a=0
        if self.mouseLButton is True and self.mouseRButton is True:
            tb.zoom_to(dx, -dy)
        elif self.mouseRButton is True:
            tb.trans_to(dx, -dy)
        elif self.mouseLButton is True:
            tb.drag_to(x, y, dx, -dy)
        self.mouseLastPos = np.array([x, y])

    def passiveMotionFunc(self, x, y):
        if self.curInteractorIX is not None:
            self.interactors[self.curInteractorIX].passiveMotion(x, y)
            return

        self.mouseLastPos = np.array([x, y])
        #print("self.mouseLastPos = " + str(self.mouseLastPos))
            
    def mykeyboard(self, key, x, y):
        #regardless of the interactor conext, 'm' always switches contexts
        keycode = ord(key)
        self.key_down[keycode] = True #set the keydown variable
        #print(keycode)
        if keycode == 46: #'>'
            try:
                self.env.currentController = min(len(self.env.controllers)-1, self.env.currentController+1)
                self.env.controllers[self.env.currentController].setup()
                self.env.stepsSinceControlSwitch = 0
                print("Switched to " + str(self.env.controllers[self.env.currentController].name))
                if self.env.save_state_on_control_switch:
                    fname = self.env.state_save_directory + self.env.controllers[self.env.currentController].name
                    print(fname)
                    count = 0
                    objfname_ix = fname+"%05d"%count
                    charfname_ix = fname+"_char%05d"%count
                    while os.path.isfile(objfname_ix+".obj"):
                        count += 1
                        objfname_ix = fname + "%05d"%count
                        charfname_ix = fname+"_char%05d"%count
                    print(objfname_ix)
                    self.env.saveObjState(filename=objfname_ix)
                    self.env.saveCharacterState(filename=charfname_ix)
                    print("...successfully saved state")
            except:
                print("no controllers to switch")
            return
        if keycode == 44: #'<'
            try:
                self.env.currentController = max(0, self.env.currentController-1)
                self.env.controllers[self.env.currentController].setup()
                self.env.stepsSinceControlSwitch = 0
                print("Switched to " + str(self.env.controllers[self.env.currentController].name))
            except:
                print("no controllers to switch")
            return
        if keycode == 115: #'s'
            try:
                print("trying to save state")
                #fname = self.env.state_save_directory + self.env.controllers[self.env.currentController].name
                fname = self.env.state_save_directory + "matchgrip"
                print(fname)
                count = 0
                objfname_ix = fname + "%05d" % count
                charfname_ix = fname + "_char%05d" % count
                while os.path.isfile(objfname_ix + ".obj"):
                    count += 1
                    objfname_ix = fname + "%05d" % count
                    charfname_ix = fname + "_char%05d" % count
                print(objfname_ix)
                self.env.saveObjState(filename=objfname_ix)
                self.env.saveCharacterState(filename=charfname_ix)
                print("...successfully saved state")
            except:
                print("...could not save the state")
            return
        if keycode == 83: #'S'
            try:
                self.env.saveCharacterRenderState()
            except:
                print("failed to save render state")
            return
        if keycode == 109: #'m'
            self.switchInteractorContext()
            return
        if keycode == 27: #esc
            self.close()
            return
        if keycode == 32: #space bar
            if self.env is not None:
                self.env.simulating = not self.env.simulating
            #self.is_simulating = not self.is_simulating
            return
        if keycode == 114: #'r' #reset
            if self.resetFunc is not None:
                self.resetFunc()
            return
        #if an interactor context is defined, pass control to it
        if self.curInteractorIX is not None:
            self.interactors[self.curInteractorIX].keyboard(key, x, y)
            return
        if keycode == 99: #'c' hijack capture
            self.capturing = not self.capturing
            print("self.capturing: " + str(self.capturing))
            #self.captureToFile(directory="/home/alexander/Documents/frame_capture_output")
            return
        if keycode == 102: #'f' apply force to garment particles
            force = np.array([1.0,1.0,1.0])
            force = np.random.uniform(low=-1, high=1, size=3)
            force /= np.linalg.norm(force)
            force *= 0.5 #scale
            if self.env is not None:
                numParticles = self.env.clothScene.getNumVertices(cid=0)
                forces = np.zeros(numParticles*3)
                for i in range(numParticles):
                    forces[i*3] = force[0]
                    forces[i*3 + 1] = force[1]
                    forces[i*3 + 2] = force[2]
                self.env.clothScene.addAccelerationToParticles(cid=0, a=forces)
                print("applied " + str(force) + " force to cloth.")

            return
        if keycode == 112: #'p'
            print("Camera data:")
            print("trans: " + str(self.scene.tb.trans))
            print("orientation(deg): " + str(self.scene.tb._get_orientation()))
            #print("orientation(deg): " + str(self.scene.tb._get_orientation()[0]*180/math.pi) + ", " + str(self.scene.tb._get_orientation()[1]*180/math.pi))
            print("rotation: " + str(self.scene.tb._rotation))
            print("matrix: " + str(self.scene.tb._matrix))
            print("zoom: " + str(self.scene.tb.zoom))
            print("distance: " + str(self.scene.tb.distance))
            #print("trans: " + str(self.scene.tb.trans))
            #print("trans: " + str(self.scene.tb.trans))

        if keycode == 106: #'j' joint constraint test
            try:
                print(hasattr(self.env, 'graphJointConstraintViolation'))
                self.env.graphJointConstraintViolation()
            except:
                print("Graph Joint Constraint Violation did not work out. Sorry.")

        #if no interactor context, do the following
        if keycode == 13: #ENTER
            if self.inputFunc is not None:
                self.inputFunc()
        self.keyPressed(key, x, y)

    def keyboardUp(self, key, x, y):
        keycode = ord(key)
        self.key_down[keycode] = False  # set the keydown variable
        if self.curInteractorIX is not None:
            self.interactors[self.curInteractorIX].keyboardUp(key, x, y)
            return

    def unproject(self, winp=np.array([0.,0.,0.])):
        #unproject the given input window space cords and return an object space vector
        obj = GLU.gluUnProject(winp[0], winp[1], winp[2], self.modelviewM, self.projectionM, self.viewport)
        return obj

    def project(self, worldp=np.array([0.,0.,0.])):
        winp = GLU.gluProject(worldp[0], worldp[1], worldp[2], self.modelviewM, self.projectionM, self.viewport)
        return winp

    def drawSphere(self, p, r, solid=True, slices=10):
        GL.glPushMatrix()
        GL.glTranslated(p[0], p[1], p[2])
        if solid is True:
            GLUT.glutSolidSphere(r, slices, slices)
        else:
            GLUT.glutWireSphere(r, slices, slices)
        GL.glPopMatrix()

    def drawLine(self, p0=np.array([0.,0,0]), p1=np.array([0.,0,0])):
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3d(p0[0],p0[1],p0[2])
        GL.glVertex3d(p1[0],p1[1],p1[2])
        GL.glEnd()

    def switchInteractorContext(self, ix=-1):
        #switch to the interactor in the list with ix or next one if ix=-1
        self.lastContextSwitch = 0
        if len(self.interactors) == 0:
            self.curInteractorIX = None
            return
        if ix == -1:
            if self.curInteractorIX is None:
                self.curInteractorIX = 0
            else:
                self.curInteractorIX = self.curInteractorIX + 1
                if self.curInteractorIX >= len(self.interactors):
                    self.curInteractorIX = None
        elif ix >= len(self.interactors):
            self.curInteractorIX = None
        else:
            self.curInteractorIX = ix

#Define interactor objects
#Each Interactor defines a set of methods which allow the user input (keyboard, mouse, etc) to change the scene
class BaseInteractor(object):
    #base interactor defining class standard methods
    def __init__(self, viewer):
        self.viewer = viewer
        self.label = "Base Interactor"

    def keyboard(self, key, x, y):
        keycode = ord(key)
        print("key down = " + str(keycode))
        if keycode == 27:
            self.viewer.close()
            return
        if keycode == 13: #ENTER
            if self.viewer.inputFunc is not None:
                self.viewer.inputFunc()
        self.viewer.keyPressed(key, x, y)

    def keyboardUp(self, key, x, y):
        keycode = ord(key)
        print("key up = " + str(keycode))

    def click(self, button, state, x, y):
        statetext = "down"
        if state == 1:
            statetext = "up"
        print("Clicked " + str(button) + " " + str(statetext) + " at ["+str(x)+","+str(y)+"]")
        if state == 0:  # Mouse pressed
            if button == 0:
                self.viewer.mouseLButton = True
            if button == 2:
                self.viewer.mouseRButton = True
            self.mouseLastPos = np.array([x, y])
            if button == 3: #mouse wheel up
                a=0
            elif button == 4: #mouse wheel down
                a=0
        elif state == 1: #mouse released
            #self.mouseLastPos = None
            if button == 0:
                self.viewer.mouseLButton = False
            if button == 2:
                self.viewer.mouseRButton = False

    def passiveMotion(self, x, y):
        self.viewer.mouseLastPos = np.array([x, y])

    def drag(self, x, y):
        dx = x - self.mouseLastPos[0]
        dy = y - self.mouseLastPos[1]
        self.viewer.mouseLastPos = np.array([x, y])

    def contextRender(self):
        #place any extra rendering for this context here
        a=0

class VertexSelectInteractor(BaseInteractor):
    #vertrex selector interactor unprojection and vertex selection
    def __init__(self, viewer):
        self.viewer = viewer
        self.label = "Vertex Select Interactor"
        self.v_down = False
        self.t_down = False
        self.x_down = False
        self.y_down = False
        self.z_down = False
        self.h_toggle = False #handle continuity toggle. When True, the handle_node will not auto destroy on mouse release
        self.selectedVertex = None
        self.selectedVertices = []
        self.handleNode = None

    def keyboard(self, key, x, y):
        keycode = ord(key)

        if keycode == 27:
            self.viewer.close()
            return
        if keycode == 13: #ENTER
            if self.viewer.inputFunc is not None:
                self.viewer.inputFunc()
        if keycode == 70: #'F'
            if self.viewer.clothScene.cullFrontFaces:
                self.viewer.clothScene.cullFrontFaces = False
                self.viewer.clothScene.cullBackFaces = True
            elif self.viewer.clothScene.cullBackFaces:
                self.viewer.clothScene.cullFrontFaces = False
                self.viewer.clothScene.cullBackFaces = False
            else:
                self.viewer.clothScene.cullFrontFaces = True
                self.viewer.clothScene.cullBackFaces = False
        if keycode == 97: #'a'
            if self.selectedVertex is not None:
                if self.selectedVertex not in self.selectedVertices:
                    self.selectedVertices.append(self.selectedVertex)
            return
        if keycode == 98:  # 'b'
            self.viewer.clothScene.renderClothBoundary = not self.viewer.clothScene.renderClothBoundary
            print("renderClothBoundary = " + str(self.viewer.clothScene.renderClothBoundary))
        if keycode == 99:  # 'c'
            self.selectedVertices = []
            return
        if keycode == 102: #'f'
            self.viewer.clothScene.renderClothFill = not self.viewer.clothScene.renderClothFill
            print("renderClothFill = " + str(self.viewer.clothScene.renderClothFill))
        if keycode == 104: #'h'
            self.h_toggle = not self.h_toggle
            if self.h_toggle is False:
                self.destroyHandleNode()
        if keycode == 112:  # 'p'
            #print relevant info
            print("Selected Vertex = " + str(self.selectedVertex))
            print("Selected Vertices = " + str(self.selectedVertices))
        if keycode == 116: #'t'
            self.t_down = True
        if keycode == 118: #'v'
            self.v_down = True
        if keycode == 119: #'w'
            self.viewer.clothScene.renderClothWires = not self.viewer.clothScene.renderClothWires
            print("renderClothWires = " + str(self.viewer.clothScene.renderClothWires))
        if keycode == 120:  # 'x'
            self.x_down = True
        if keycode == 121:  # 'y'
            self.y_down = True
        if keycode == 122:  # 'z'
            self.z_down = True
        self.viewer.keyPressed(key, x, y)

    def keyboardUp(self, key, x, y):
        keycode = ord(key)
        if keycode == 116: #'t'
            self.t_down = False
        if keycode == 118: #'v'
            self.v_down = False
        if keycode == 120:  # 'x'
            self.x_down = False
        if keycode == 121:  # 'y'
            self.y_down = False
        if keycode == 122:  # 'z'
            self.z_down = False

    def createHandleNodeFromSelected(self):
        #weed out already pinned verts to avoid double pinning
        newhandleverts = []
        for v in self.selectedVertices:
            if not self.viewer.clothScene.getPinned(cid=0, vid=v):
                newhandleverts.append(v)

        #also look at the currently selected vert
        if self.selectedVertex is not None:
            if not self.viewer.clothScene.getPinned(cid=0, vid=self.selectedVertex):
                newhandleverts.append(self.selectedVertex)

        if len(newhandleverts) > 0:
            #create a HandleNode from any valid selected verts
            self.handleNode = HandleNode(clothScene=self.viewer.clothScene)


            self.handleNode.addVertices(verts=newhandleverts)
            self.handleNode.setOrgToCentroid()

    def destroyHandleNode(self):
        #destroy the selected HandleNode
        if self.handleNode is not None:
            self.handleNode.clearHandles()
        self.handleNode = None

    def click(self, button, state, x, y):
        tb = self.viewer.scene.tb
        if state == 0:  # Mouse pressed
            if self.t_down or self.x_down or self.y_down or self.z_down:
                if self.handleNode is None:
                    self.createHandleNodeFromSelected()
            if button == 0:
                self.viewer.mouseLButton = True
                if self.v_down is True:
                    self.selectedVertex = self.viewer.clothScene.getCloseVertex(self.viewer.prevWorldMouse)
                    if self.selectedVertex not in self.selectedVertices:
                        self.selectedVertices.append(self.selectedVertex)
            if button == 2:
                self.viewer.mouseRButton = True
                if self.v_down is True:
                    self.selectedVertex = self.viewer.clothScene.getCloseVertex(self.viewer.prevWorldMouse)
                    if self.selectedVertex is not None:
                        print("selected vertex " + str(self.selectedVertex) + " pos = " + str(self.viewer.clothScene.getVertexPos(cid=0, vid=self.selectedVertex)))
            self.viewer.mouseLastPos = np.array([x, y])
            if button == 3: #wheel up
                if self.v_down is True:
                    if self.selectedVertex is not None:
                        self.selectedVertex += 1
                        if self.selectedVertex >= self.viewer.clothScene.getNumVertices():
                            self.selectedVertex = 0
                        elif self.selectedVertex < 0:
                            self.selectedVertex = self.viewer.clothScene.getNumVertices()-1
                    else:
                        self.selectedVertex = 0
                elif self.t_down is True:
                    self.handleNode.org = self.handleNode.org + self.viewer.camForward * 0.05
                    self.handleNode.updateHandles()
                else:
                    tb.trans[2] += 0.1
            elif button == 4: #wheel down
                if self.v_down is True:
                    if self.selectedVertex is not None:
                        self.selectedVertex -= 1
                        if self.selectedVertex >= self.viewer.clothScene.getNumVertices():
                            self.selectedVertex = 0
                        elif self.selectedVertex < 0:
                            self.selectedVertex = self.viewer.clothScene.getNumVertices()-1
                    elif self.t_down is True:
                        self.handleNode.org = self.handleNode.org - self.viewer.camForward * 0.05
                        self.handleNode.updateHandles()
                    else:
                        self.selectedVertex = self.selectedVertex = self.viewer.clothScene.getNumVertices()-1
                else:
                    tb.trans[2] -= 0.1
        elif state == 1: #mouse released
            if self.h_toggle is False:
                self.destroyHandleNode()
            #self.mouseLastPos = None
            if button == 0:
                self.viewer.mouseLButton = False
            if button == 2:
                self.viewer.mouseRButton = False

    def drag(self, x, y):
        dx = x - self.viewer.mouseLastPos[0]
        dy = y - self.viewer.mouseLastPos[1]

        if self.v_down is True:
            a=0
        elif self.t_down is True:
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a = 0
            elif self.viewer.mouseRButton is True:
                # compute a rotation about camUp and camRight axiis
                self.handleNode.applyAxisAngle(dx * (6.28 / 360.), self.viewer.camForward)
                self.handleNode.applyAxisAngle(dy * (6.28 / 360.), self.viewer.camRight)
                #self.handle_node.updateHandles()
            elif self.viewer.mouseLButton is True:
                # translate the frame org with camUp and camRight
                self.handleNode.org = self.handleNode.org + dx * self.viewer.camRight * 0.001
                self.handleNode.org = self.handleNode.org + dy * self.viewer.camUp * -0.001
                #self.handle_node.updateHandles()
        elif self.x_down is True:
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a=0
            elif self.viewer.mouseLButton is True:
                rightX = np.array([1., 0, 0])
                if self.viewer.camRight.dot(rightX) < 0:
                    rightX *= -1
                self.handleNode.org = self.handleNode.org + dx * rightX * 0.001
                #self.handle_node.updateHandles()
            elif self.viewer.mouseRButton is True:
                #rotation about x
                self.handleNode.applyLocalAxisAngle(dy*(6.28/360.), np.array([1.,0,0]))
                #self.handle_node.updateHandles()
        elif self.y_down is True:
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a=0
            elif self.viewer.mouseLButton is True:
                upY = np.array([0, 1., 0])
                if self.viewer.camUp.dot(upY) > 0:
                    upY *= -1
                self.handleNode.org = self.handleNode.org + dy * upY * 0.001
                #self.handle_node.updateHandles()
            elif self.viewer.mouseRButton is True:
                #rotation about y
                self.handleNode.applyLocalAxisAngle(dx * (6.28 / 360.), np.array([0., 1., 0]))
                #self.handle_node.updateHandles()
        elif self.z_down is True:
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a=0
            elif self.viewer.mouseLButton is True:
                forwardZ = np.array([0, 0., 1.])
                if self.viewer.camForward.dot(forwardZ) > 0:
                    forwardZ *= -1
                self.handleNode.org = self.handleNode.org + dy * forwardZ * 0.001
                #self.handle_node.updateHandles()
            elif self.viewer.mouseRButton is True:
                # rotation about z
                self.handleNode.applyLocalAxisAngle(dx * (6.28 / 360.), np.array([0., 0., 1.]))
                #self.handle_node.updateHandles()
        else:
            tb = self.viewer.scene.tb
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                tb.zoom_to(dx, -dy)
            elif self.viewer.mouseRButton is True:
                tb.trans_to(dx, -dy)
            elif self.viewer.mouseLButton is True:
                tb.drag_to(x, y, dx, -dy)

        self.viewer.mouseLastPos = np.array([x, y])

    def contextRender(self):
        #place any extra rendering for this context here
        if self.v_down is True:
            GL.glColor3d(1,0,0)
        self.viewer.drawSphere(p=self.viewer.prevWorldMouse, r=0.001, solid=True)

        if self.handleNode is not None:
            self.handleNode.updateHandles()
            self.handleNode.draw()
        if self.h_toggle is True:
            self.viewer.clothScene.drawText(x=self.viewer.viewport[2] / 2, y=self.viewer.viewport[3] - 60, text="Handle Node is toggled On", color=(0., 0, 0))

        GL.glColor3d(0, 1, 0)
        for v in self.selectedVertices:
            #print("v = " + str(v))
            vx = self.viewer.clothScene.getVertexPos(vid=v)
            self.viewer.drawSphere(p=vx, r=0.003, solid=True)

        selectedVertexText = "None"
        if self.selectedVertex is not None:
            vx = self.viewer.clothScene.getVertexPos(vid=self.selectedVertex)
            GL.glColor3d(1, 1, 0)
            self.viewer.drawSphere(p=vx, r=0.002, solid=True)
            selectedVertexText = str(self.selectedVertex)
        self.viewer.clothScene.drawText(x=self.viewer.viewport[2] / 2, y=self.viewer.viewport[3] - 45, text="Selected Vertex = " + str(selectedVertexText), color=(0., 0, 0))
        a=0

class FrameSelectInteractor(BaseInteractor):
    # base interactor defining class standard methods
    def __init__(self, viewer):
        self.viewer = viewer
        self.label = "Frame Select Interactor"
        self.f_down = False
        self.x_down = False
        self.y_down = False
        self.z_down = False
        self.frame = pyutils.ShapeFrame(org=np.array([0.,0,-1.75]))

    def key_down(self):
        #check for all function key_down booleans
        if self.f_down is True:
            return True
        if self.x_down is True:
            return True
        if self.y_down is True:
            return True
        if self.z_down is True:
            return True
        return False

    def keyboard(self, key, x, y):
        keycode = ord(key)
        if keycode == 27:
            self.viewer.close()
            return
        if keycode == 13:  # ENTER
            if self.viewer.inputFunc is not None:
                self.viewer.inputFunc()
        if keycode == 102:  # 'f'
            self.f_down = True
        if keycode == 112:  # 'p'
            # print relevant info
            print("Frame T = " + str(self.frame.org))
            print("Frame Q = " + str(self.frame.quat))
            print("Frame R = " + str(self.frame.orientation))
        if keycode == 120:  # 'x'
            self.x_down = True
        if keycode == 121:  # 'y'
            self.y_down = True
        if keycode == 122:  # 'z'
            self.z_down = True
        self.viewer.keyPressed(key, x, y)

    def keyboardUp(self, key, x, y):
        keycode = ord(key)
        if keycode == 102:  # 'f'
            self.f_down = False
        if keycode == 120:  # 'x'
            self.x_down = False
        if keycode == 121:  # 'y'
            self.y_down = False
        if keycode == 122:  # 'z'
            self.z_down = False

    def click(self, button, state, x, y):
        tb = self.viewer.scene.tb
        if state == 0:  # Mouse pressed
            if button == 0:
                self.viewer.mouseLButton = True
            if button == 2:
                self.viewer.mouseRButton = True
            self.viewer.mouseLastPos = np.array([x, y])
            if button == 3:  # wheel up
                if self.f_down is True:
                    self.frame.org = self.frame.org + self.viewer.camForward * 0.1
                else:
                    tb.trans[2] += 0.1
            elif button == 4:  # wheel down
                if self.f_down is True:
                    self.frame.org = self.frame.org + self.viewer.camForward * -0.1
                else:
                    tb.trans[2] -= 0.1
        elif state == 1:  # mouse released
            # self.mouseLastPos = None
            if button == 0:
                self.viewer.mouseLButton = False
            if button == 2:
                self.viewer.mouseRButton = False

    def drag(self, x, y):
        dx = x - self.viewer.mouseLastPos[0]
        dy = y - self.viewer.mouseLastPos[1]

        if self.key_down() is False:
            tb = self.viewer.scene.tb
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                tb.zoom_to(dx, -dy)
            elif self.viewer.mouseRButton is True:
                tb.trans_to(dx, -dy)
            elif self.viewer.mouseLButton is True:
                tb.drag_to(x, y, dx, -dy)
        elif (self.f_down):
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a=0
            elif self.viewer.mouseRButton is True:
                #compute a rotation about camUp and camRight axiis
                self.frame.applyAxisAngle(dx * (6.28 / 360.), self.viewer.camForward)
                self.frame.applyAxisAngle(dy * (6.28 / 360.), self.viewer.camRight)
                a=0
            elif self.viewer.mouseLButton is True:
                #translate the frame org with camUp and camRight
                self.frame.org = self.frame.org + dx * self.viewer.camRight * 0.001
                self.frame.org = self.frame.org + dy * self.viewer.camUp * -0.001
        elif (self.x_down):
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a=0
            elif self.viewer.mouseLButton is True:
                rightX = np.array([1., 0, 0])
                if self.viewer.camRight.dot(rightX) < 0:
                    rightX *= -1
                self.frame.org = self.frame.org + dx * rightX * 0.001
            elif self.viewer.mouseRButton is True:
                #rotation about x
                self.frame.applyLocalAxisAngle(dy*(6.28/360.), np.array([1.,0,0]))
        elif (self.y_down):
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a=0
            elif self.viewer.mouseLButton is True:
                upY = np.array([0, 1., 0])
                if self.viewer.camUp.dot(upY) > 0:
                    upY *= -1
                self.frame.org = self.frame.org + dy * upY * 0.001
            elif self.viewer.mouseRButton is True:
                #rotation about y
                self.frame.applyLocalAxisAngle(dx * (6.28 / 360.), np.array([0., 1., 0]))
        elif (self.z_down):
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                a=0
            elif self.viewer.mouseLButton is True:
                forwardZ = np.array([0, 0., 1.])
                if self.viewer.camForward.dot(forwardZ) > 0:
                    forwardZ *= -1
                self.frame.org = self.frame.org + dy * forwardZ * 0.001
            elif self.viewer.mouseRButton is True:
                # rotation about z
                self.frame.applyLocalAxisAngle(dx * (6.28 / 360.), np.array([0., 0., 1.]))

        self.viewer.mouseLastPos = np.array([x, y])

    def contextRender(self):
        # place any extra rendering for this context here
        GL.glColor3d(0, 1, 0)
        if self.f_down is True:
            GL.glColor3d(1, 0, 0)
        self.frame.draw()

        a = 0

class IKInteractor(BaseInteractor):
    # base interactor defining class standard methods
    def __init__(self, viewer):
        self.viewer = viewer
        self.label = "Inverse Kinematics Select Interactor"
        self.f_down = False
        self.x_down = False
        self.y_down = False
        self.z_down = False
        self.s_down = False #s_down for handle select
        #inverse kinematics info
        self.ikTargets = []
        self.ikOffsets = []
        self.ikNodes = []
        self.selectedHandle = None
        self.skelix = 1 #change this if the skel file changes

    def key_down(self):
        #check for all function key_down booleans
        if self.f_down is True:
            return True
        if self.s_down is True:
            return True
        if self.x_down is True:
            return True
        if self.y_down is True:
            return True
        if self.z_down is True:
            return True
        return False

    def keyboard(self, key, x, y):
        keycode = ord(key)
        if keycode == 27:
            self.viewer.close()
            return
        if keycode == 13:  # ENTER
            if self.viewer.inputFunc is not None:
                self.viewer.inputFunc()
        if keycode == 102:  # 'f'
            self.f_down = True
        if keycode == 112:  # 'p'
            # print relevant info
            for i in range(len(self.ikNodes)):
                print("IK Handle " + str(i) + " | node " + str(self.ikNodes[i]) + " | offset " + str(self.ikOffsets[i]) + " | target " + str(self.ikTargets[i]))
        if keycode == 115:  # 's'
            self.s_down = True
        if keycode == 120:  # 'x'
            self.x_down = True
        if keycode == 121:  # 'y'
            self.y_down = True
        if keycode == 122:  # 'z'
            self.z_down = True
        self.viewer.keyPressed(key, x, y)

    def keyboardUp(self, key, x, y):
        keycode = ord(key)
        if keycode == 102:  # 'f'
            self.f_down = False
        if keycode == 115:  # 's'
            self.s_down = False
        if keycode == 120:  # 'x'
            self.x_down = False
        if keycode == 121:  # 'y'
            self.y_down = False
        if keycode == 122:  # 'z'
            self.z_down = False

    def drag(self, x, y):
        dx = x - self.viewer.mouseLastPos[0]
        dy = y - self.viewer.mouseLastPos[1]
        if self.key_down() is False:
            tb = self.viewer.scene.tb
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                tb.zoom_to(dx, -dy)
            elif self.viewer.mouseRButton is True:
                tb.trans_to(dx, -dy)
            elif self.viewer.mouseLButton is True:
                tb.drag_to(x, y, dx, -dy)
        elif (self.f_down):
            if self.selectedHandle is not None:
                disp = dx * self.viewer.camRight * 0.001 + dy * self.viewer.camUp * -0.001
                if self.viewer.mouseLButton is True:
                    self.ikTargets[self.selectedHandle] += disp
                elif self.viewer.mouseRButton is True:
                    skel = self.viewer.sim.skeletons[self.skelix]
                    node = skel.bodynodes[self.Nodes[self.selectedHandle]]
                    self.ikOffsets[self.selectedHandle] = node.to_local(node.to_world(self.ikOffsets[self.selectedHandle]) + disp)
        elif (self.x_down):
            if self.selectedHandle is not None:
                self.ikOffsets[self.selectedHandle][0] += dx*0.001
        elif (self.y_down):
            if self.selectedHandle is not None:
                self.ikOffsets[self.selectedHandle][1] += dx*0.001
        elif (self.z_down):
            if self.selectedHandle is not None:
                self.ikOffsets[self.selectedHandle][2] += dx*0.001

        self.viewer.mouseLastPos = np.array([x, y])

    def click(self, button, state, x, y):
        tb = self.viewer.scene.tb
        if state == 0:  # Mouse pressed
            if button == 0:
                self.viewer.mouseLButton = True
                if self.s_down:
                    close = None
                    close_dist = 999
                    thresh = 0.5
                    skel = self.viewer.sim.skeletons[self.skelix]
                    for ix, n in enumerate(skel.bodynodes):
                        dist = np.linalg.norm(n.com()-self.viewer.prevWorldMouse)
                        if dist < close_dist and dist < thresh:
                            close_dist = dist
                            close = ix
                    if close is not None:
                        create = True
                        for ix,n in enumerate(self.ikNodes):
                            if n == close:
                                create = False
                                self.selectedHandle = ix
                                break
                        if create:
                            #create new handle
                            self.ikNodes.append(close)
                            self.ikOffsets.append(skel.bodynodes[close].to_local(self.viewer.prevWorldMouse))
                            self.ikTargets.append(self.viewer.prevWorldMouse)
                            self.selectedHandle = len(self.ikNodes)-1
            if button == 2:
                self.viewer.mouseRButton = True
                if self.s_down:
                    close = None
                    close_dist = 999
                    thresh = 0.5
                    skel = self.viewer.sim.skeletons[self.skelix]
                    for ix, n in enumerate(skel.bodynodes):
                        dist = np.linalg.norm(n.com() - self.viewer.prevWorldMouse)
                        if dist < close_dist and dist < thresh:
                            close_dist = dist
                            close = ix
                    if close is not None:
                        for ix,n in enumerate(self.ikNodes):
                            if n == close:
                                del self.ikTargets[ix]
                                del self.ikOffsets[ix]
                                del self.ikNodes[ix]
                            break
                        self.selectedHandle = None
            self.viewer.mouseLastPos = np.array([x, y])
            if button == 3:  # wheel up
                if self.f_down is True:
                    self.ikTargets[self.selectedHandle] += 0.01*self.viewer.camForward
                elif self.s_down is True:
                    if len(self.ikNodes) == 0:
                        self.selectedHandle = None
                    elif self.selectedHandle is None:
                        self.selectedHandle = 0
                    else:
                        self.selectedHandle += 1
                        if self.selectedHandle >= len(self.ikNodes):
                            self.selectedHandle = None
                else:
                    tb.trans[2] += 0.1
            elif button == 4:  # wheel down
                if self.f_down is True:
                    self.ikTargets[self.selectedHandle] -= 0.01 * self.viewer.camForward
                elif self.s_down is True:
                    if len(self.ikNodes) == 0:
                        self.selectedHandle = None
                    elif self.selectedHandle is None:
                        self.selectedHandle = len(self.ikNodes)-1
                    else:
                        self.selectedHandle -= 1
                        if self.selectedHandle < 0:
                            self.selectedHandle = None
                else:
                    tb.trans[2] -= 0.1
        elif state == 1:  # mouse released
            # self.mouseLastPos = None
            if button == 0:
                self.viewer.mouseLButton = False
            if button == 2:
                self.viewer.mouseRButton = False

    def contextRender(self):
        # place any extra rendering for this context here
        for i in range(len(self.ikNodes)):
            skel = self.viewer.sim.skeletons[self.skelix]
            handlepos = skel.bodynodes[self.ikNodes[i]].to_world(self.ikOffsets[i])
            GL.glColor3d(0, 1, 1)
            self.viewer.drawSphere(p=handlepos, r=0.025, solid=True)
            GL.glColor3d(1, 0, 1)
            self.viewer.drawSphere(p=self.ikTargets[i], r=0.025, solid=True)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3d(self.ikTargets[i][0], self.ikTargets[i][1], self.ikTargets[i][2])
            GL.glVertex3d(handlepos[0], handlepos[1], handlepos[2])
            GL.glEnd()


        #draw selected handle info
        if self.selectedHandle is None:
            self.viewer.clothScene.drawText(x=self.viewer.viewport[2] / 2, y=self.viewer.viewport[3] - 45, text="Selected Handle = " + str(self.selectedHandle), color=(0., 0, 0))
        else:
            skel = self.viewer.sim.skeletons[self.skelix]
            node_name = skel.bodynodes[self.ikNodes[self.selectedHandle]].name
            self.viewer.clothScene.drawText(x=self.viewer.viewport[2] / 2, y=self.viewer.viewport[3] - 45,
                                            text="Selected Handle = " + str(self.selectedHandle) + " | " + str(node_name) + " | offset= " + str(self.ikOffsets[self.selectedHandle]) + " | target = " + str(self.ikTargets[self.selectedHandle]), color=(0., 0, 0))
        a = 0

class PoseInteractor(BaseInteractor):
    # interactor defining GUI influence over skeleton DOFS by clicking displayed DOF boxes
    def __init__(self, viewer):
        self.viewer = viewer
        self.label = "Pose DOF Interactor"
        self.boxRanges = [] #list of np array pairs: [[minx, maxx],[miny, maxy]]
        self.sawyerBoxRanges = [] #list of np array pairs: [[minx, maxx],[miny, maxy]]
        self.skelix = 1  # change this if the skel file changes
        self.selectedBox = None
        self.selectedSkel = None
        self.boxesDefined = False
        self.topLeft = None
        self.sawyerTopLeft = None
        self.selectedNode = None
        self.selectedNodeOffset = np.zeros(3)
        self.clickz = 0
        self.mouseGlobal = None
        self.mouseAtClick = np.zeros(2)

    def defineBoxRanges(self):
        #manually defined
        #These values are taken from pyPhysX.renderUtils.renderDofs()
        skel = self.viewer.sim.skeletons[self.skelix]
        #if self.topLeft is None:
        self.topLeft = np.array([2., self.viewer.viewport[3] - 10])
        textWidth = 165.  # pixel width of the text
        numWidth = 50.  # pixel width of the lower/upper bounds text
        barWidth = 90.
        barHeight = -15.
        barSpace = -5.
        #print("topLeft " + str(topLeft))
        self.boxRanges = []
        for i in range(len(skel.q)):
            self.boxRanges.append([[self.topLeft[0] + textWidth + numWidth, self.topLeft[0] + textWidth + numWidth + barWidth], [self.topLeft[1] + (barHeight + barSpace) * i + barHeight, self.topLeft[1] + (barHeight + barSpace) * i]])
        #print(self.boxRanges)

        try:
            sawyer_skel = self.viewer.env.sawyer_skel

            self.sawyerTopLeft = np.array([75, self.viewer.viewport[3]-450])
            self.sawyerBoxRanges = []
            barWidth = 120.
            barHeight = -16.
            barSpace = -4.
            for i in range(len(sawyer_skel.q)-6):
                self.sawyerBoxRanges.append(
                    [[self.sawyerTopLeft[0], self.sawyerTopLeft[0] + barWidth],
                     [self.sawyerTopLeft[1] + (barHeight + barSpace) * i + barHeight,
                      self.sawyerTopLeft[1] + (barHeight + barSpace) * i]])

        except:
            a = 0
            try:
                iiwa_skel = self.viewer.env.iiwa_skel

                self.sawyerTopLeft = np.array([75, self.viewer.viewport[3] - 450])
                self.sawyerBoxRanges = []
                barWidth = 120.
                barHeight = -16.
                barSpace = -4.
                for i in range(len(iiwa_skel.q) - 6):
                    self.sawyerBoxRanges.append(
                        [[self.sawyerTopLeft[0], self.sawyerTopLeft[0] + barWidth],
                         [self.sawyerTopLeft[1] + (barHeight + barSpace) * i + barHeight,
                          self.sawyerTopLeft[1] + (barHeight + barSpace) * i]])

            except:
                a = 0


    def boxClickTest(self, point):
        print(point)
        if not self.boxesDefined:
            self.defineBoxRanges()
            self.boxesDefined = True
        #return None if nothing hit, or the box/dof index of the clicked box
        point[1] = self.viewer.viewport[3] - point[1]
        for ix,b in enumerate(self.boxRanges):
            if point[0] < b[0][1] and point[0] > b[0][0]:
                if point[1] < b[1][1] and point[1] > b[1][0]:
                    #print("clicked box " + str(ix))
                    self.selectedSkel = self.viewer.env.robot_skeleton
                    return ix

        #if not in a skel box, test the sawyer boxes
        for ix,b in enumerate(self.sawyerBoxRanges):
            if point[0] < b[0][1] and point[0] > b[0][0]:
                if point[1] < b[1][1] and point[1] > b[1][0]:
                    #print("clicked sawyer box " + str(ix))
                    try:
                        self.selectedSkel = self.viewer.env.sawyer_skel
                    except:
                        self.selectedSkel = self.viewer.env.iiwa_skel
                    return ix + 6

        return None

    def incrementSelectedDOF(self, inc):
        #increment a selected dof by inc respecting joint limits
        if self.selectedBox is not None:
            #skel = self.viewer.sim.skeletons[self.skelix]
            skel = self.selectedSkel
            qpos = skel.q
            qpos[self.selectedBox] += inc
            if not math.isinf(skel.position_lower_limits()[self.selectedBox]) and not math.isinf(skel.position_upper_limits()[self.selectedBox]):
                qpos[self.selectedBox] = min(max(qpos[self.selectedBox], skel.position_lower_limits()[self.selectedBox]), skel.position_upper_limits()[self.selectedBox])
            skel.set_positions(qpos)

    def distToSkel(self, medial=False):
        #return the distance from the previous mouse position (3D) to the skeleton and the nearest bodynode
        skel = self.viewer.sim.skeletons[self.skelix]
        self.selectedNode = None
        self.selectedNodeOffset = np.zeros(3)
        sphereInfo = self.viewer.clothScene.getCollisionSpheresInfo()
        capsuleInfo = self.viewer.clothScene.getCollisionCapsuleInfo()
        capsuleBodynodes = self.viewer.clothScene.collisionCapsuleBodynodes
        if capsuleBodynodes is None:
            print("No bodynode correspondance present in clothscene. Aborting.")
            return
        smallestDistance = 9999
        bestTangentialDistance = 0
        closestNode = -1
        bestLine = None
        for row in range(len(capsuleInfo)):
            for col in range(len(capsuleInfo)):
                if capsuleInfo[row][col] != 0:
                    #print("capsule between spheres " + str(row) + " and " + str(col))
                    p0 = sphereInfo[9*row:9*row+3]
                    r0 = sphereInfo[9 * row + 3]
                    p1 = sphereInfo[9*col:9*col+3]
                    r1 = sphereInfo[9 * col + 3]
                    #print("length = " + str(np.linalg.norm(p0-p1)))
                    distance, tangentialDistance = pyutils.distToLine(p=self.viewer.prevWorldMouse, l0=p0, l1=p1, distOnLine=True) #note: distances returned in p0->p1 space (not unit space)
                    radius = r0 + (r1-r0)*tangentialDistance
                    if distance-radius < smallestDistance:
                        smallestDistance = distance-radius
                        bestTangentialDistance = tangentialDistance
                        closestNode = int(capsuleBodynodes[row][col])
                        bestLine = (p0, p1, tangentialDistance)
        if smallestDistance < 0.01:
            self.selectedNode = closestNode
            if not medial: #grab a point on the surface of the node
                self.selectedNodeOffset = skel.bodynodes[closestNode].to_local(self.viewer.prevWorldMouse)
            else: #grab a point on the medial axis of the node
                nodeDirection = (bestLine[1]-bestLine[0])
                #nodeDirection /= np.linalg.norm(nodeDirection)
                self.selectedNodeOffset = skel.bodynodes[closestNode].to_local(bestLine[0]+nodeDirection*bestLine[2])
            print("Clicked " + str(skel.bodynodes[closestNode]) + " at distance " + str(smallestDistance) + " and extent " + str(bestTangentialDistance))

    def click(self, button, state, x, y):
        tb = self.viewer.scene.tb
        self.mouseAtClick = np.array([x,y])
        if state == 0:  # Mouse pressed
            self.selectedBox = self.boxClickTest(np.array([x,y]))
            if not self.viewer.mouseRButton and not self.viewer.mouseLButton:
                self.clickz = GL.glReadPixels(self.viewer.mouseLastPos[0],
                                            self.viewer.viewport[3] - self.viewer.mouseLastPos[1], 1, 1,
                                            GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
                self.distToSkel(medial=(button == 2))
            if button == 0:
                self.viewer.mouseLButton = True
            if button == 2:
                self.viewer.mouseRButton = True
            self.viewer.mouseLastPos = np.array([x, y])
            if button == 3:  # wheel up
                if self.selectedBox is not None:
                    self.incrementSelectedDOF(0.05)
                else:
                    tb.trans[2] += 0.1
            elif button == 4:  # wheel down
                if self.selectedBox is not None:
                    self.incrementSelectedDOF(-0.05)
                else:
                    tb.trans[2] -= 0.1
            elif button == 7: #front side mouse button
                self.clickz -= 0.0001
                if self.selectedNode is not None:
                    self.mouseGlobal = np.array(self.viewer.unproject(np.array(
                        [self.viewer.mouseLastPos[0], self.viewer.viewport[3] - self.viewer.mouseLastPos[1],
                         self.clickz])))
            elif button == 8: #back side mouse button
                self.clickz += 0.0001
                if self.selectedNode is not None:
                    self.mouseGlobal = np.array(self.viewer.unproject(np.array(
                        [self.viewer.mouseLastPos[0], self.viewer.viewport[3] - self.viewer.mouseLastPos[1],
                         self.clickz])))
        elif state == 1:  # mouse released
            self.selectedBox = None
            self.viewer.env.supplementalTau *= 0.0
            # self.mouseLastPos = None
            if button == 0:
                self.viewer.mouseLButton = False
            if button == 2:
                self.viewer.mouseRButton = False
            if not self.viewer.mouseRButton and not self.viewer.mouseLButton:
                self.selectedNode = None
                self.mouseGlobal = None
                self.selectedNodeOffset = np.zeros(3)

    def drag(self, x, y):
        dx = x - self.viewer.mouseLastPos[0]
        dy = y - self.viewer.mouseLastPos[1]

        if self.selectedNode is not None:
            self.mouseGlobal = np.array(self.viewer.unproject(np.array([self.viewer.mouseLastPos[0], self.viewer.viewport[3] - self.viewer.mouseLastPos[1], self.clickz])))
            forceScale = 20.0
            skel = self.viewer.sim.skeletons[self.skelix]
            #skel.bodynodes[self.selectedNode].add_ext_force(_force=forceScale*(self.viewer.camRight * dx + self.viewer.camUp * -dy), _offset=self.selectedNodeOffset)
            #skel.bodynodes[self.selectedNode].add_ext_force(_force=forceScale*(self.mouseGlobal - skel.bodynodes[self.selectedNode].to_world(self.selectedNodeOffset)), _offset=self.selectedNodeOffset)

        elif self.selectedBox is None:
            tb = self.viewer.scene.tb
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                tb.zoom_to(dx, -dy)
            elif self.viewer.mouseRButton is True:
                tb.trans_to(dx, -dy)
            elif self.viewer.mouseLButton is True:
                tb.drag_to(x, y, dx, -dy)
        else:
            skel = self.viewer.sim.skeletons[self.skelix]
            if self.viewer.mouseLButton:
                if not math.isinf(skel.position_lower_limits()[self.selectedBox]) and not math.isinf(skel.position_upper_limits()[self.selectedBox]):
                    dofRange = skel.position_upper_limits()[self.selectedBox]-skel.position_lower_limits()[self.selectedBox]
                    self.incrementSelectedDOF(dx*((dofRange)/(self.boxRanges[self.selectedBox][0][1]-self.boxRanges[self.selectedBox][0][0])))
                else:
                    self.incrementSelectedDOF(dx * 0.05)
            else:
                if self.viewer.env is not None:
                    self.viewer.env.supplementalTau[self.selectedBox] = (self.viewer.mouseLastPos[0] - self.mouseAtClick[0])*0.1
        self.viewer.mouseLastPos = np.array([x, y])

    def keyboard(self, key, x, y):
        keycode = ord(key)
        if keycode == 27:
            self.viewer.close()
            return
        if keycode == 13:  # ENTER
            if self.viewer.inputFunc is not None:
                self.viewer.inputFunc()
            return
        if keycode == 98: #'b'
            self.defineBoxRanges()
            return
        if keycode == 112:  # 'p'
            # print relevant info
            skel = self.viewer.sim.skeletons[self.skelix]
            dofstr = "["
            for ix,dof in enumerate(skel.q):
                dofstr = dofstr + str(dof)
                if ix < len(skel.q)-1:
                    dofstr = dofstr + ", "
            dofstr = dofstr + "]"
            print(dofstr)
            return
        if keycode == 80:  # 'P'
            print("Trying to save pose and cloth state")
            try:
                self.viewer.env.saveObjState()
                self.viewer.env.saveCharacterState()
            except:
                print("failed to save obj file or character state")
            return
        if keycode == 118: # 'v'
            #0 character velocity to stop drift
            print("zeroing velocity")
            skel = self.viewer.sim.skeletons[self.skelix]
            vpos = np.zeros(skel.ndofs)
            skel.set_velocities(vpos)
        self.viewer.keyPressed(key, x, y)

    def contextRender(self):
        if self.selectedBox is not None and self.viewer.mouseRButton:
            #applying offset distance based torque to a joint, draw the offset lines
            mouseAtClickCorrected = np.array([self.mouseAtClick[0], self.viewer.viewport[3]-self.mouseAtClick[1]])
            mouseLastCorrected = np.array([self.viewer.mouseLastPos[0], self.viewer.viewport[3]-self.viewer.mouseLastPos[1]])
            yVector = np.array([mouseLastCorrected[0], mouseAtClickCorrected[1]])
            renderUtils.drawLines2D(points=[mouseAtClickCorrected, mouseLastCorrected, mouseAtClickCorrected, yVector])

        if self.mouseGlobal is not None and self.selectedNode is not None:
            skel = self.viewer.sim.skeletons[self.skelix]
            self.viewer.drawSphere(p=self.mouseGlobal, r=0.01, solid=True)
            handlePos = skel.bodynodes[self.selectedNode].to_world(self.selectedNodeOffset)
            forceDir = self.mouseGlobal-handlePos
            forceMag = np.linalg.norm(forceDir)
            forceDir /= forceMag
            forceMag = min(1.0, forceMag)**2
            renderUtils.drawSphere(pos=handlePos,rad=0.02)
            renderUtils.setColor(color=renderUtils.heatmapColor(minimum=0,maximum=1.0,value=forceMag))
            renderUtils.drawLineStrip(points=[handlePos, self.mouseGlobal])
            renderUtils.drawArrow(p0=handlePos, p1=handlePos+forceDir*0.2,hwRatio=0.15)
            renderUtils.drawExtendedAxis(self.mouseGlobal)
            forceScale = 10000.0
            skel.bodynodes[self.selectedNode].add_ext_force(_force=forceScale*forceDir*forceMag, _offset=self.selectedNodeOffset)
        elif np.linalg.norm(self.viewer.prevWorldMouse-self.viewer.camForwardNear) > 0.05:
            self.viewer.drawSphere(p=self.viewer.prevWorldMouse, r=0.01, solid=True)

        if self.selectedBox is not None:
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glPushMatrix()
            GL.glLoadIdentity()
            GL.glOrtho(0, self.viewer.viewport[2], 0, self.viewer.viewport[3], -1, 1)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glPushMatrix()
            GL.glLoadIdentity()
            GL.glDisable(GL.GL_CULL_FACE)

            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glColor3d(3., 3., 0)
            GL.glBegin(GL.GL_QUADS)
            b = self.boxRanges[self.selectedBox]
            #for b in self.boxRanges:
            GL.glVertex2d(b[0][0]-1, b[1][0]-1)
            GL.glVertex2d(b[0][0]-1, b[1][1]+1)
            GL.glVertex2d(b[0][1]+1, b[1][1]+1)
            GL.glVertex2d(b[0][1]+1, b[1][0]-1)
            GL.glEnd()
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glMatrixMode(GL.GL_PROJECTION)
            GL.glPopMatrix()
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glPopMatrix()

class Frame6DInteractor(BaseInteractor):
    # controls for manual manipulation of 6dof frame
    def __init__(self, viewer):
        self.viewer = viewer
        self.label = "Frame Interactor"
        self.rotationToggle = False
        self.f_down = False
        self.q_down = False
        self.x_down = False
        self.y_down = False
        self.z_down = False
        self.s_down = False #s_down for handle select
        #inverse kinematics info
        self.ikTargets = []
        self.ikOffsets = []
        self.ikNodes = []
        self.selectedHandle = None
        self.skelix = 1 #change this if the skel file changes

    def key_down(self):
        #check for all function key_down booleans
        if self.f_down is True:
            return True
        if self.s_down is True:
            return True
        if self.x_down is True:
            return True
        if self.y_down is True:
            return True
        if self.z_down is True:
            return True
        if self.q_down is True:
            return True
        return False

    def keyboard(self, key, x, y):
        keycode = ord(key)
        #print(keycode)
        if keycode == 27:
            self.viewer.close()
            return
        if keycode == 13:  # ENTER
            if self.viewer.inputFunc is not None:
                self.viewer.inputFunc()
        if keycode == 102:  # 'f'
            self.f_down = True
        if keycode == 112:  # 'p'
            # print relevant info
            #print()
            a=0
        if keycode == 113:  # 'q'
            self.q_down = True
        if keycode == 115:  # 's'
            self.s_down = True
        if keycode == 116:  # 't'
            self.rotationToggle = not self.rotationToggle
            print("rotation toggle: " + str(self.rotationToggle))

        if keycode == 104:  # 'h' hand
            self.viewer.env.frameInterpolator["target_pos"] = self.viewer.env.robot_skeleton.bodynodes[11].to_world(np.zeros(3))

        if keycode == 101:  # 'e' elbow
            self.viewer.env.frameInterpolator["target_pos"] = self.viewer.env.robot_skeleton.bodynodes[10].to_world(np.zeros(3))

        if keycode == 118: #'v' shoulder
            self.viewer.env.frameInterpolator["target_pos"] = self.viewer.env.robot_skeleton.bodynodes[9].to_world(np.zeros(3))

        if keycode == 120:  # 'x'
            self.x_down = True
        if keycode == 121:  # 'y'
            self.y_down = True
        if keycode == 122:  # 'z'
            self.z_down = True
        self.viewer.keyPressed(key, x, y)

    def keyboardUp(self, key, x, y):
        keycode = ord(key)
        if keycode == 102:  # 'f'
            self.f_down = False
        if keycode == 113:  # 'q'
            self.q_down = False
        if keycode == 115:  # 's'
            self.s_down = False
        if keycode == 120:  # 'x'
            self.x_down = False
        if keycode == 121:  # 'y'
            self.y_down = False
        if keycode == 122:  # 'z'
            self.z_down = False

    def drag(self, x, y):
        dx = x - self.viewer.mouseLastPos[0]
        dy = y - self.viewer.mouseLastPos[1]
        if self.key_down() is False:
            tb = self.viewer.scene.tb
            if self.viewer.mouseLButton is True and self.viewer.mouseRButton is True:
                tb.zoom_to(dx, -dy)
            elif self.viewer.mouseRButton is True:
                tb.trans_to(dx, -dy)
            elif self.viewer.mouseLButton is True:
                tb.drag_to(x, y, dx, -dy)
        elif (self.f_down):
            a=0
        elif (self.q_down):
            self.viewer.env.armTargetPathInfo["target_angle"] += dx*0.001
            print("target_angle = " + str(self.viewer.env.armTargetPathInfo["target_angle"]))
        elif (self.x_down):
            if not self.rotationToggle:
                try:
                    self.viewer.env.frameInterpolator["target_pos"][0] += dx*0.001
                except:
                    self.viewer.env.iiwas[0].frameInterpolator["target_pos"][0] += dx*0.001
            else:
                self.viewer.env.frameInterpolator["eulers"][0] += dx*0.001
                #self.viewer.env.frameEulerState[0] += dx*0.001
        elif (self.y_down):
            if not self.rotationToggle:
                try:
                    self.viewer.env.frameInterpolator["target_pos"][1] += dx * 0.001
                except:
                    self.viewer.env.iiwas[0].frameInterpolator["target_pos"][1] += dx*0.001
            else:
                self.viewer.env.frameInterpolator["eulers"][1] += dx*0.001
                #self.viewer.env.frameEulerState[1] += dx*0.001
        elif (self.z_down):
            if not self.rotationToggle:
                try:
                    self.viewer.env.frameInterpolator["target_pos"][2] += dx * 0.001
                except:
                    self.viewer.env.iiwas[0].frameInterpolator["target_pos"][2] += dx*0.001
            else:
                self.viewer.env.frameInterpolator["eulers"][2] += dx * 0.001
                #self.viewer.env.frameEulerState[2] += dx * 0.001

        self.viewer.mouseLastPos = np.array([x, y])

    def click(self, button, state, x, y):
        tb = self.viewer.scene.tb
        if state == 0:  # Mouse pressed
            if button == 0:
                self.viewer.mouseLButton = True

            if button == 2:
                self.viewer.mouseRButton = True

            self.viewer.mouseLastPos = np.array([x, y])

            if button == 3:  # wheel up
                if self.f_down is True:
                    a = 0
                elif self.s_down is True:
                    a = 0
                else:
                    tb.trans[2] += 0.1
            elif button == 4:  # wheel down
                if self.f_down is True:
                    a = 0
                elif self.s_down is True:
                    a = 0
                else:
                    tb.trans[2] -= 0.1
        elif state == 1:  # mouse released
            # self.mouseLastPos = None
            if button == 0:
                self.viewer.mouseLButton = False
            if button == 2:
                self.viewer.mouseRButton = False

    def contextRender(self):
        # place any extra rendering for this context here

        a = 0
