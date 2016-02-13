'use strict';

// requirements

var gulp = require('gulp'),
    browserify = require('gulp-browserify'),
    size = require('gulp-size'),
    less = require('gulp-less'),
    rename = require('gulp-rename'),
    clean = require('gulp-clean');

gulp.task('jsx', function () {
  return gulp.src('./project/static/scripts/jsx/main.jsx')
    .pipe(browserify({
        insertGlobals: true,
        extensions: ['.jsx'],
        transform: ['reactify']
    }))
    .pipe(rename({extname: '.js'}))
    .pipe(gulp.dest('./project/static/scripts/generated_js'))
    .pipe(size())
});

gulp.task('less', function () {
  return gulp.src('./project/static/styles/less/*.less')
  .pipe(less())
  // .pipe(cssmin())
  // .pipe(rename({suffix: '.min'}))
  .pipe(gulp.dest('./project/static/styles/generated_css'));
});

gulp.task('clean', function () {
  return gulp.src(
      ['./project/static/scripts/generated_js',
       './project/static/styles/generated_css'],
      {read: false})
    .pipe(clean());
});

gulp.task('watch', function () {
  gulp.watch('./project/static/scripts/jsx/*', ['jsx']);
  gulp.watch('./project/static/styles/less/*', ['less']);
});

gulp.task('default', ['clean', 'jsx', 'less', 'watch']);
