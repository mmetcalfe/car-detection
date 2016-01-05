import OpenGL.GL as gl

def glGenBuffers(num):
    print 'glGenBuffers', num
    return gl.glGenBuffers(num)
def glDeleteBuffers(num, buffs):
    print 'glDeleteBuffers', num, buffs
    gl.glDeleteBuffers(num, buffs)
def glBindBuffer(target, id):
    print 'glBindBuffer', target, id
    gl.glBindBuffer(target, id)
def glBufferData(target, size, data, usage):
    print 'glBufferData', target, size, data, usage
    gl.glBufferData(target, size, data, usage)
def glGenVertexArrays(num):
    print 'glGenVertexArrays', num
    return gl.glGenVertexArrays(num)
def glDeleteVertexArrays(num, buffs):
    print 'glDeleteVertexArrays', num, buffs
    gl.glDeleteVertexArrays(num, buffs)
def glBindVertexArray(id):
    print 'glBindVertexArray', id
    gl.glBindVertexArray(id)
def glEnableVertexAttribArray(loc):
    print 'glEnableVertexAttribArray', loc
    gl.glEnableVertexAttribArray(loc)
def glVertexAttribPointer(location,size,type,normalized,stride,offsetPtr):
    print 'glVertexAttribPointer', location,size,type,normalized,stride,offsetPtr
    gl.glVertexAttribPointer(location,size,type,normalized,stride,offsetPtr)
def glCreateShader(shaderType):
    print 'glCreateShader', shaderType
    return gl.glCreateShader(shaderType)
def glDeleteShader(id):
    print 'glDeleteShader', id
    gl.glDeleteShader(id)
def glCompileShader(id):
    print 'glCompileShader', id
    gl.glCompileShader(id)
def glShaderSource(id, sources):
    print 'glShaderSource', id, sources
    gl.glShaderSource(id, sources)
def glCreateProgram():
    print 'glCreateProgram',
    return gl.glCreateProgram()
def glDeleteProgram(id):
    print 'glDeleteProgram', id
    gl.glDeleteProgram(id)
def glAttachShader(progId, shadId):
    print 'glAttachShader', progId, shadId
    gl.glAttachShader(progId, shadId)
def glBindFragDataLocation(id, colorNumber, name):
    print 'glBindFragDataLocation', id, colorNumber, name
    gl.glBindFragDataLocation(id, colorNumber, name)
def glLinkProgram(id):
    print 'glLinkProgram', id
    gl.glLinkProgram(id)
def glUseProgram(id):
    print 'glUseProgram', id
    gl.glUseProgram(id)
def glValidateProgram(id):
    print 'glValidateProgram', id
    gl.glValidateProgram(id)
def glGetUniformLocation(id, name):
    print 'glGetUniformLocation', id, name
    return gl.glGetUniformLocation(id, name)
def glGetProgramiv(id, name):
    print 'glGetProgramiv', id, name
    return gl.glGetProgramiv(id, name)
def glGetProgramInfoLog(id):
    print 'glGetProgramInfoLog', id
    return gl.glGetProgramInfoLog(id)
def glFlush():
    print 'glFlush'
    gl.glFlush()
def glDrawElements(typ, num, dtyp, off):
    print 'glDrawElements', typ, num, dtyp, off
    gl.glDrawElements(typ, num, dtyp, off)
def glDrawArrays(typ, off, count):
    print 'glDrawArrays', typ, off, count
    gl.glDrawArrays(typ, off, count)
def glBindFramebuffer(typ, num):
    print 'glBindFramebuffer', typ, num
    gl.glBindFramebuffer(typ, num)
def glViewport(a,b,c,d):
    print 'glViewport', a,b,c,d
    gl.glViewport(a,b,c,d)
def glClearColor(r,g,b,a):
    print 'glClearColor', r,g,b,a
    gl.glClearColor(r,g,b,a)
def glClear(chnl):
    print 'glClear', chnl
    gl.glClear(chnl)
def glEnable(name):
    print 'glEnable', name
    gl.glEnable(name)
def glDetachShader(progId, shadId):
    print 'glDetachShader', progId, shadId
    gl.glDetachShader(progId, shadId)
def glGetAttribLocation(id, name):
    print 'glGetAttribLocation', id, name
    return gl.glGetAttribLocation(id, name)
def glUniform1f(loc, val):
    print 'glUniform1f', loc, val
    gl.glUniform1f(loc, val)
def glUniformMatrix4fv(a, b,c,d):
    print 'glUniformMatrix4fv', a, b,c,d
    gl.glUniformMatrix4fv(a, b,c,d)
