import os

from OpenGL.GL import *
# from opengldebug import *
import numpy as np

import pyassimp as imp
from pprint import pprint

from camera import *

# void Model::setCameraUniformsOnShaderPrograms(Camera &camera, glm::mat4 model) {
#     if (textureProgram != nullptr) {
#         setCameraUniformsOnShaderProgram(textureProgram, camera, model);
#     }
#
#     if (flatProgram != nullptr) {
#         setCameraUniformsOnShaderProgram(flatProgram, camera, model);
#     }
#
#     if (environmentMapProgram != nullptr) {
#         setCameraUniformsOnShaderProgram(environmentMapProgram, camera, model);
#     }
# }
#
# void Model::setCameraUniformsOnShaderProgram(std::shared_ptr<NUGL::ShaderProgram> program, Camera &camera, glm::mat4 model) {
#     glm::mat4 mvp = camera.proj * camera.view * model;
#
#     program->use();
#     program->setUniformIfActive("model", model);
#     program->setUniformIfActive("view", camera.view);
#     program->setUniformIfActive("proj", camera.proj);
#
#     program->setUniformIfActive("mvp", mvp);
#
#     // We don't invert the transforms relating to the model's internal structure.
#     glm::mat4 modelViewInverse = glm::inverse(camera.view * transform);
#     program->setUniformIfActive("modelViewInverse", modelViewInverse);
#
#     glm::mat4 viewInverse = glm::inverse(camera.view);
#     program->setUniformIfActive("viewInverse", viewInverse);
# }

def getSizeOfOpenGlType(type):
    if type == GL_BYTE:
        return 1 # sizeof(GLbyte);
    elif type == GL_UNSIGNED_BYTE:
        return 1 # sizeof(GLubyte);
    elif type == GL_SHORT:
        return 2 # sizeof(GLshort);
    elif type == GL_UNSIGNED_SHORT:
        return 2 # sizeof(GLushort);
    elif type == GL_INT:
        return 4 # sizeof(GLint);
    elif type == GL_UNSIGNED_INT:
        return 4 # sizeof(GLuint);
    elif type == GL_FLOAT:
        return 4 # sizeof(GLfloat);
    elif type == GL_DOUBLE:
        return 8 # sizeof(GLdouble);
    else:
        raise ValueError("Unknown type enum value " + type + ".")

class Buffer:
    def __init__(self):
        self.id = glGenBuffers(1)
        self.count = 0

    def delete(self):
        glDeleteBuffers(1, [self.id])

    def bind(self, target):
        glBindBuffer(target, self.id)

    def setData(self, target, data, usage):
        self.bind(target)
        glBufferData(target, data.nbytes, data, usage)
        self.count = len(data)

class VertexAttribute:
    def __init__(self, name, size, typ, normalized, isPadding):
        self.name = name
        self.size = size
        self.type = typ
        self.normalized = normalized
        self.isPadding = isPadding

class VertexArray:
    def __init__(self):
        self.id = glGenVertexArrays(1)

    def delete(self):
        glDeleteVertexArrays(1, [self.id])

    def bind(self):
        glBindVertexArray(self.id)

    # Assumes sequential, packed vertices.
    #setAttributePointers :: ShaderProgram -> Buffer -> GLenum -> [VertexAttribute]) {
    def setAttributePointers(self, program, buffer, target, attribs):
        stride = 0
        for attrib in attribs:
            stride += attrib.size * getSizeOfOpenGlType(attrib.type)

        self.bind()

        offset = 0
        for attrib in attribs:
            if not attrib.isPadding:
                location = program.getAttribLocation(attrib.name)

                glEnableVertexAttribArray(location)

                buffer.bind(target)

                offsetPtr = ctypes.c_void_p(offset)
                glVertexAttribPointer(
                        location,
                        attrib.size,
                        attrib.type,
                        attrib.normalized,
                        stride,
                        offsetPtr)

            offset += attrib.size * getSizeOfOpenGlType(attrib.type)

class Shader:
    def __init__(self, shaderType):
        self.id = glCreateShader(shaderType)
        self.type = shaderType
        self.sourceFile = ''

    def delete(self):
        glDeleteShader(self.id)

    def compile(self):
        glCompileShader(self.id);

    def setSource(self, source):
        glShaderSource(self.id, [source])
        self.sourceFile = None

    def setSourceFromFile(self, fname):
        with open(fname, 'r') as fs:
            source = fs.read()
            self.setSource(source)
            self.sourceFile = fname

    @classmethod
    def fromFile(cls, shaderType, fname):
        shader = cls(shaderType)
        shader.setSourceFromFile(fname)
        shader.compile()
        return shader
    @classmethod
    def fromString(cls, shaderType, source):
        shader = cls(shaderType)
        shader.setSource(source)
        shader.compile()
        return shader

    def printDebugInfo(self):
        compileStatus = glGetShaderiv(self.id, GL_COMPILE_STATUS);
        shaderType = glGetShaderiv(self.id, GL_SHADER_TYPE);
        deleteStatus = glGetShaderiv(self.id, GL_DELETE_STATUS);
        sourceLength = glGetShaderiv(self.id, GL_SHADER_SOURCE_LENGTH);
        infoLogLength = glGetShaderiv(self.id, GL_INFO_LOG_LENGTH);
        infoLogBuff = glGetShaderInfoLog(self.id);
        print "printShaderDebugInfo(" + str(self.id) + ") {"
        print "  GL_SHADER_TYPE: "
        if self.type == GL_VERTEX_SHADER:
            print "GL_VERTEX_SHADER"
        elif self.type == GL_FRAGMENT_SHADER:
            print "GL_FRAGMENT_SHADER"
        else:
            print "INVALID (" + str(self.type) + ")"
        print ","
        print "  GL_DELETE_STATUS: " + ("Flagged for deletion" if deleteStatus else "False")
        print "  GL_COMPILE_STATUS: " + ("Success" if compileStatus else "Failure")
        print "  GL_INFO_LOG_LENGTH: " + str(infoLogLength)
        print "  GL_SHADER_SOURCE_LENGTH: " + str(sourceLength)
        print "  glGetShaderInfoLog: " + infoLogBuff
        print "}"

        if not compileStatus:
            raise ValueError(": Compile error in shader: " + infoLogBuff + ".")


class ShaderProgram:
    def __init__(self, name, shaders):
        self.id = glCreateProgram()
        self.name = name
        self.attached = {}
        for sh in shaders:
            self.attachShader(sh)

    def delete(self):
        glDeleteProgram(self.id)
        for sh in self.attached.values():
            sh.delete()

    def attachShader(self, shader):
        assert(not (shader.type in self.attached))

        self.attached[shader.type] = shader
        glAttachShader(self.id, shader.id);

    def bindFragDataLocation(self, colorNumber, name):
        glBindFragDataLocation(self.id, colorNumber, name)

    def link(self):
        glLinkProgram(self.id)
    def use(self):
        glUseProgram(self.id)
    def validate(self):
        glValidateProgram(self.id)

    #     uniLoc = glGetUniformLocation(self.id, name)
    def uniformLocation(self, name):
        return glGetUniformLocation(self.id, name)

    def getAttribLocation(self, name):
        return glGetAttribLocation(self.id, name)

    def setUniformMat4(self, name, value, transpose = True):
        self.use()

        uniLoc = self.uniformLocation(name)

        mat = np.array(value.flatten(), np.float32)

        glUniformMatrix4fv(uniLoc, 1, transpose, mat)

    # def uniformIsActive(self, name):
    #     uniLoc = glGetUniformLocation(self.id, name)
    #     return (uniLoc != -1)
    # def setUniformIfActive(self, name, value):
    #     if self.uniformIsActive(name):
    #         self.setUniform(name, value)

    @classmethod
    def fromFiles(cls, name, vertShaderPath, fragShaderPath):
        vertShader = Shader.fromFile(GL_VERTEX_SHADER, vertShaderPath)
        fragShader = Shader.fromFile(GL_FRAGMENT_SHADER, fragShaderPath)

        program = cls(name, [vertShader, fragShader])

        return program
    @classmethod
    def fromStrings(cls, name, vertShaderSource, fragShaderSource):
        vertShader = Shader.fromString(GL_VERTEX_SHADER, vertShaderSource)
        fragShader = Shader.fromString(GL_FRAGMENT_SHADER, fragShaderSource)

        program = cls(name, [vertShader, fragShader])

        return program

    def printDebugInfo(self):
        for sh in self.attached.values():
            sh.printDebugInfo()

        deleteStatus = glGetProgramiv(self.id, GL_DELETE_STATUS)
        linkStatus = glGetProgramiv(self.id, GL_LINK_STATUS)
        validateStatus = glGetProgramiv(self.id, GL_VALIDATE_STATUS)
        infoLogLength = glGetProgramiv(self.id, GL_INFO_LOG_LENGTH)
        attachedShaders = glGetProgramiv(self.id, GL_ATTACHED_SHADERS)
        activeAttributes = glGetProgramiv(self.id, GL_ACTIVE_ATTRIBUTES)
        activeAttributeMaxLength = glGetProgramiv(self.id, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH)
        activeUniforms = glGetProgramiv(self.id, GL_ACTIVE_UNIFORMS)
        activeUniformMaxLength = glGetProgramiv(self.id, GL_ACTIVE_UNIFORM_MAX_LENGTH)
        infoLogBuff = glGetProgramInfoLog(self.id)
        print "printProgramDebugInfo(" + str(self.id) + ", " + self.name +  ") {"
        print "  GL_DELETE_STATUS: " + ("Flagged for deletion" if deleteStatus else "False") + ","
        print "  GL_LINK_STATUS: " + ("Success" if linkStatus else "Failure") + ","
        print "  GL_VALIDATE_STATUS: " + ("Success" if validateStatus else "Failure") + ","
        print "  GL_INFO_LOG_LENGTH: " + str(infoLogLength) + ","
        print "  GL_ATTACHED_SHADERS: " + str(attachedShaders) + ","
        print "  GL_ACTIVE_ATTRIBUTES: " + str(activeAttributes) + ","
        print "  GL_ACTIVE_ATTRIBUTE_MAX_LENGTH: " + str(activeAttributeMaxLength) + ","
        print "  GL_ACTIVE_UNIFORMS: " + str(activeUniforms) + ","
        print "  GL_ACTIVE_UNIFORM_MAX_LENGTH: " + str(activeUniformMaxLength) + ","
        print "  glGetProgramInfoLog: " + str(infoLogBuff) + ","
        print "}"

        if linkStatus != True:
            raise ValueError("Link error in shader program " + str(self.name) + "': " + str(infoLogBuff))

class MeshBuffer:

    @staticmethod
    def facesToElements(faces):
        return np.array([i for fc in faces for i in fc], np.uint32)

    def __init__(self, mesh):
        self.vertex = None
        self.element = None
        # Map shader programs to vertex arrays:
        self.vertexArrayMap = {}

        vertexData = []
        for i in range(0, len(mesh.vertices)):
            vertexData.extend(mesh.vertices[i])
            vertexData.extend(mesh.normals[i])
            # vertexData.extend(mesh.texturecoords[i][:2])

        vertexData = np.array(vertexData, np.float32)
        self.vertex = Buffer()
        self.vertex.setData(GL_ARRAY_BUFFER, vertexData, GL_STATIC_DRAW)

        elements = MeshBuffer.facesToElements(mesh.faces)
        self.element = Buffer()
        self.element.setData(GL_ELEMENT_ARRAY_BUFFER, elements, GL_STATIC_DRAW)

    def delete(self):
        self.vertex.delete()
        self.element.delete()

        for arr in self.vertexArrayMap.values():
            arr.delete()

    def prepareVertexArrayForShaderProgram(self, program):
        attribs = [
            # name, size, typ, normalized, isPadding
            VertexAttribute(
              name = "position"
            , size = 3
            , typ = GL_FLOAT
            , normalized = False
            , isPadding = False)
            ,
            VertexAttribute(
              name = "normal"
            , size = 3
            , typ = GL_FLOAT
            , normalized = False
            , isPadding = False)
        ]

        # if (hasNormals()):
        #     attribs.push_back({"normal", 3, GL_FLOAT, GL_FALSE, !program->attributeIsActive("normal")});

        program.use()

        vertexArray = VertexArray()
        vertexArray.bind()
        vertexArray.setAttributePointers(program, self.vertex, GL_ARRAY_BUFFER, attribs)

        self.vertexArrayMap[program] = vertexArray

    def draw(self, program):
        # self.prepareMaterialShaderProgram(program);
        # program.setUniform("colAmbient", material->colAmbient);
        # location = program.uniformLocation("colAmbient")

        # Mesh::draw
        program.use();
        if not program in self.vertexArrayMap.keys():
            prepareVertexArrayForShaderProgram(program);

        self.vertexArrayMap[program].bind()
        self.vertex.bind(GL_ARRAY_BUFFER) # Not necessary?
        self.element.bind(GL_ELEMENT_ARRAY_BUFFER)

        # glDrawArrays(GL_TRIANGLES, 0, self.vertex.count)
        # glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        glDrawElements(GL_TRIANGLES, self.element.count, GL_UNSIGNED_INT, ctypes.c_void_p(0))


class Model:
    def __init__(self, fname):
        self.aiModel = imp.load(fname)

        self.meshBuffers = []

        # defaults:
        self.dir = np.array([1, 0, 0], np.float32)
        self.up = np.array([0, 0, 1], np.float32)
        self.pos = np.array([0, 0, 0], np.float32)
        self.scale = np.array([1, 1, 1], np.float32)

    def release(self):
        imp.release(self.aiModel)

        for buff in self.meshBuffers:
            buff.delete()

    def generateBuffers(self):
        assert(len(self.meshBuffers) == 0)
        for mesh in self.aiModel.meshes:
            self.meshBuffers.append(MeshBuffer(mesh))

    def prepareVertexArraysForShaderProgram(self, program):
        for buff in self.meshBuffers:
            buff.prepareVertexArrayForShaderProgram(program)

    def draw(self, program, model=None):
        scale = np.eye(4, dtype=np.float32)
        scale[0,0] = self.scale[0]
        scale[1,1] = self.scale[1]
        scale[2,2] = self.scale[2]

        if model == None:
            orient = lookAtTransform(self.pos, self.pos + self.dir, self.up, square=True)
            # model = np.linalg.inv(orient)*scale
            model = np.linalg.inv(orient)*scale
        else:
            orient = lookAtTransform(self.pos, self.pos + self.dir, self.up, square=True)
            # model = model*np.linalg.inv(orient)*scale
            model = model*np.linalg.inv(orient)*scale
        program.setUniformMat4('model', model)

        # for mesh in self.aiModel.meshes:
        #     for i in range(0, len(mesh.vertices)):
        #         # print 'model', model
        #         vert = np.row_stack([np.matrix(mesh.vertices[i]).T, np.array([1])])
        #         worldVert = model*vert
        #         eyeVert = proj*worldVert
        #         ndcVert = eyeVert[:3]/eyeVert[3]
        #         print 'worldVert:', worldVert.T
        #         # print '   m->w', worldVert.T
        #         print '   w->e', eyeVert.T
        #         print '   e->n', ndcVert.T

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        for mesh in self.meshBuffers:
            mesh.draw(program)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

class Scene:
    def __init__(self):
        self.models = {}

        vertShaderPath = os.path.join(os.path.dirname(__file__), 'glsl/position.vert')
        fragShaderPath = os.path.join(os.path.dirname(__file__), 'glsl/uniform.frag')
        flatProgram = ShaderProgram.fromFiles('flat', vertShaderPath, fragShaderPath)
        flatProgram.bindFragDataLocation(0, "outColor")
        flatProgram.link()
        flatProgram.validate()
        flatProgram.printDebugInfo()
        self.flatProgram = flatProgram

        modelPath = os.path.join(os.path.dirname(__file__), 'models/cube.obj')
        cubeModel = Model(modelPath)
        cubeModel.generateBuffers()
        cubeModel.prepareVertexArraysForShaderProgram(flatProgram)
        cubeModel.pos = np.array([-1, 1, 4], np.float32)
        cubeModel.dir = np.array([0, 0, -1], np.float32)
        cubeModel.up = np.array([0, 1, 0], np.float32)
        cubeModel.scale = np.array([1, 1, 1], np.float32)
        self.models['cube'] = cubeModel

    def release(self):
        self.flatProgram.delete()

        for model in self.models.values():
            model.release()

    def prepareFrame(self, camera):
        # renderTest()
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        #
        # # glViewport(0, 0, windowSize[0], windowSize[1]);
        # glViewport(0, 0, windowSize[0], windowSize[1]);
        # w, h = windowSize
        # glViewport(0, 0, w, h);

        glClearColor(0, 0, 0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);
        # glClear(GL_COLOR_BUFFER_BIT)

        # # Backface culling:
        # glEnable(GL_CULL_FACE); # cull face
        # glCullFace(GL_BACK); # cull back face
        # glFrontFace(GL_CCW); # GL_CCW for counter clock-wise

        proj = camera.getOpenGlCameraMatrix()
        self.flatProgram.setUniformMat4('proj', proj)

    def renderModels(self, camera):
        self.prepareFrame(camera)

        for model in self.models.values():
            model.draw(self.flatProgram)
