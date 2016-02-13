'use strict';

// requirements

var gulp = require('gulp'),
    browserify = require('gulp-browserify'),
    size = require('gulp-size'),
    rename = require('gulp-rename'),
    clean = require('gulp-clean');


gulp.task('transform', function () {
  return gulp.src('./project/static/scripts/jsx/main.jsx')
    .pipe(browserify({transform: ['reactify']}))
    .pipe(rename({extname: '.js'}))
    .pipe(gulp.dest('./project/static/scripts/generated_js'))
    .pipe(size());
});

gulp.task('clean', function () {
  return gulp.src(['./project/static/scripts/generated_js'], {read: false})
    .pipe(clean());
});

gulp.task('default', ['clean'], function () {
  gulp.start('transform');
  gulp.watch('./project/static/scripts/jsx/main.js', ['transform']);
});
