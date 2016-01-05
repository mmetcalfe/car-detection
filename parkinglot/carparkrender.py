from carpark import *
from openglscene import *
from camera import viewPortMatrix

import cairo
from drawing2d import ExtendedCairoContext
from geometry import *

def placeModelWithinRectangle(model, rect, zPos=0, zScale=1):
    xPos, yPos, angle = rect.trans
    model.pos = np.array([xPos, yPos, zPos])

    model.dir = sphericalToCartesian([1, angle, 0])
    model.up = np.array([0, 0, 1])

    xScale, yScale = rect.size
    model.scale = np.array([xScale, yScale, zScale])


class ParkingLotRender:
    def __init__(self, parkingLot):
        self.openglScene = Scene()
        self.parkingLot = parkingLot

    def release(self):
        self.openglScene.release()

    def openglRenderCamera(self, cam, cubeModel, drawFrustum=True):
        if not drawFrustum:
            return

        cubeModel.pos = cam.pos
        cubeModel.dir = sphericalToCartesian(cam.sphericalDir)
        cubeModel.up = cam.up
        cubeModel.scale = np.array([0.2, 0.1, 0.1])
        cubeModel.draw(self.openglScene.flatProgram)

        if drawFrustum:
            cubeModel.pos = np.array([0,0,0])
            cubeModel.dir = np.array([1,0,0])
            cubeModel.up = np.array([0,0,1])
            cubeModel.scale = np.array([2,2,1])
            proj = cam.getOpenGlCameraMatrix()
            cubeModel.draw(self.openglScene.flatProgram, np.linalg.inv(proj))

    def openglDrawOnScreenDetection(self, detection, cubeModel, camera):
        # Invert OpenGL's (implicit) viewport matrix and set it as the
        # projection matrix to allow drawing directly onto the screen in units
        # of pixels.
        # Note: The viewport matrix has been set previously using:
        #    glViewport(0, 0, fbs_x, fbs_y);
        proj = np.linalg.inv(viewPortMatrix(camera.framebufferSize))
        self.openglScene.flatProgram.setUniformMat4('proj', proj)
        # print 'proj', proj

        # Draw an axis-aligned cube scaled to the size of the detection:
        cubeModel.pos = np.array([detection.trans.x, detection.trans.y, 0])
        cubeModel.dir = np.array([1, 0, 0])
        cubeModel.up = np.array([0, 0, 1])
        cubeModel.scale = np.array([detection.size[0], detection.size[1], 0.1])

        # p = camera.P*np.mat([0,0,0, 1]).T
        # print 'mul', p
        # p = p/p[2]
        # print 'div', p
        # p = camera.project(np.mat([0,0,0, 1]).T)
        # cubeModel.pos = np.array([p[0], p[1], 0])
        # cubeModel.scale = np.array([5, 5, 0.1])

        cubeModel.draw(self.openglScene.flatProgram)

        # Reset the projection matrix to the camera's projection matrix:
        proj = camera.getOpenGlCameraMatrix()
        self.openglScene.flatProgram.setUniformMat4('proj', proj)

    def getGroundProjectedDetection(self, detection, camera):
        dc = np.array([detection.trans.x, detection.trans.y, 0], np.float32)
        dhw = np.array([detection.size[0]*0.5, 0, 0])
        dhh = np.array([0, detection.size[1]*0.5, 0])
        vertices = [
            dc - dhw + dhh,
            dc + dhw + dhh,
            dc + dhw - dhh,
            dc - dhw - dhh
        ]

        vpInv = np.linalg.inv(viewPortMatrix(camera.framebufferSize))
        def unvp(u):
            x = vpInv*np.row_stack([np.mat(u).T, [1]])
            x = x[:3]
            # return np.array(x.T)
            return np.array([x[0,0], x[1,0], x[2,0]])

        vertices = [unvp(v) for v in vertices]
        gpverts = [camera.projectPointToGround(v) for v in vertices]

        return gpverts

    def openglDrawGroundProjectedDetection(self, detection, camera):
        model = Model()

        d = 0.99
        dc = np.array([detection.trans.x, detection.trans.y, d], np.float32)
        dhw = np.array([detection.size[0]*0.5, 0, 0], np.float32)
        dhh = np.array([0, detection.size[1]*0.5, 0], np.float32)
        vertices = [
            dc - dhw + dhh,
            dc + dhw + dhh,
            dc + dhw - dhh,
            dc - dhw - dhh
        ]
        normals = vertices
        faces = [
            [0, 1, 2],
            [2, 3, 0]
        ]

        # vpInv = np.linalg.inv(viewPortMatrix(camera.framebufferSize))
        # def unvp(u):
        #     x = vpInv*np.row_stack([np.mat(u).T, [1]])
        #     x = x[:3]
        #     # return np.array(x.T)
        #     return np.array([x[0,0], x[1,0], x[2,0]])
        #
        # testverts = [unvp(v) for v in vertices]
        # # print testverts
        # # testverts = [camera.unprojectOpenGL(v) for v in testverts]
        # testverts = [camera.projectPointToGround(v) for v in testverts]
        # # print 'testverts', testverts
        # vertices = testverts

        vertices = self.getGroundProjectedDetection(detection, camera)

        if not None in vertices:
            model.meshBuffers.append(MeshBuffer(vertices, normals, faces))
            model.draw(self.openglScene.flatProgram, rawVertices=True)
            model.release()

    def renderOpenGL(self, camera, playerCamera):
        self.openglScene.prepareFrame(camera)

        # Carpark config:
        mapSize = self.parkingLot.size
        mapCentre = self.parkingLot.centre

        cubeModel = self.openglScene.models['cube']

        # Draw map outline:
        trans = Transform2D.fromParts(mapSize/2-mapCentre, 0)
        rect = RotatedRectangle(trans, mapSize)
        placeModelWithinRectangle(cubeModel, rect, -0.5)
        cubeModel.draw(self.openglScene.flatProgram)

        # Draw parking spaces:
        for space in self.parkingLot.spaces:
            placeModelWithinRectangle(cubeModel, space, 0, 0.1)
            cubeModel.draw(self.openglScene.flatProgram)

        # Draw cameras:
        drawFrustum = (playerCamera!=camera)
        self.openglRenderCamera(playerCamera, cubeModel, drawFrustum=drawFrustum)
        for cam in self.parkingLot.cameras:
            drawFrustum = (cam!=camera)
            self.openglRenderCamera(cam, cubeModel, drawFrustum=drawFrustum)

        # Draw detections:
        # for detection in self.parkingLot.detections:
        #     self.openglDrawOnScreenDetection(detection, cubeModel, camera)
        #     self.openglDrawGroundProjectedDetection(detection, camera)
        cams = self.parkingLot.cameras + [playerCamera]
        for detection in self.parkingLot.detections:
            self.openglDrawOnScreenDetection(detection, cubeModel, camera)
            # self.openglDrawGroundProjectedDetection(detection, camera)

            # proj = camera.getOpenGlCameraMatrix()
            # self.openglScene.flatProgram.setUniformMat4('proj', proj)

            for cam in cams:
                self.openglDrawGroundProjectedDetection(detection, cam)


    def renderCairo(self, camera, fname):
        # Carpark config:
        mapSize = self.parkingLot.size
        mapCentre = self.parkingLot.centre

        # Drawing config:
        canvasSize = self.parkingLot.canvasSize

        # Create a canvas of the specified size:
        surface = cairo.PDFSurface(fname, canvasSize[0], canvasSize[1])
        # ctx = cairo.Context(surface)
        ctx = ExtendedCairoContext(surface)
        ctx.transformToRealWorldUnits(canvasSize, mapSize, mapCentre)

        # Draw map outline:
        ctx.set_line_width(0.05)
        ctx.setCol(ctx.getRandCol())
        trans = Transform2D.fromParts(mapSize/2-mapCentre, 0)
        rect = RotatedRectangle(trans, mapSize)
        ctx.rotatedRectangle(rect)
        ctx.stroke()

        # Draw parking spaces:
        ctx.set_line_width(0.05)
        ctx.setCol(ctx.getRandCol())
        for space in self.parkingLot.spaces:
            ctx.rotatedRectangle(space)
            ctx.stroke()
            # ctx.fill()

        # # Draw cameras:
        # ctx.set_line_width(0.01)
        # ctx.setCol(ctx.getRandCol())
        # for cam in self.parkingLot.cameras:
        #     trans2D = Transform2D.fromParts(cam.pos, cam.sphericalDir[1])
        #     ctx.rotatedRectangle(RotatedRectangle(trans2D, [0.2, 0.1]))
        #     ctx.stroke()

        ctx.set_line_width(0.05)
        ctx.setCol(ctx.getRandCol())
        for cam in self.parkingLot.cameras:
            camCol = ctx.getRandCol()
            ctx.setCol(camCol)

            # Draw cameras:
            trans2D = Transform2D.fromParts(cam.pos, cam.sphericalDir[1])
            ctx.rotatedRectangle(RotatedRectangle(trans2D, [0.2, 0.1]))
            ctx.stroke()

            # Draw projected detections:
            for detection in self.parkingLot.detections:
                pts = self.getGroundProjectedDetection(detection, cam)

                # Highlight occupied parking spaces:
                ctx.setCol(ctx.getRandCol())
                for space in self.parkingLot.spaces:
                    ctx.set_source_rgba(1, 0.8, 0.8, 0.5)
                    if intersectRectangleConvexQuad(space, pts, ctx):
                        ctx.rotatedRectangle(space)
                        ctx.fill()

                ctx.setCol(camCol)
                ctx.move_to(pts[0][0], pts[0][1])
                for pt in pts[1:]:
                    ctx.line_to(pt[0], pt[1])
                ctx.close_path()
                ctx.stroke()


        ctx.show_page()

        print 'renderCairo:', fname
