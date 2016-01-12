#
# perform_experiments.py
#
#	This script performs the entire data collection process for this project.
#	It generates trials, runs them, collates the data into a convenient format
#	for analysis, then outputs charts ready for a paper.
#

import argparse
import subprocess
import glob
import sys
import os
import re
import random
import math
from concurrent import futures

import cascadetraining as training

NUM_THREADS = 6

if __name__ == "__main__":
	random.seed(123454321) # Use deterministic samples.

	# Parse arguments:
	parser = argparse.ArgumentParser(description='Perform experiments')
	parser.add_argument('template_yaml', type=str, nargs='?', default='template.yaml', help='Filename of the YAML file describing the trials to generate.')
	parser.add_argument('output_dir', type=str, nargs='?', default='trials', help='Directory in which to output the generated trials.')
	args = parser.parse_args()

	# print '===== PREPROCESS NEGATIVE IMAGES ====='
	# print '	Create hard negative images by detecting ellipses in negative\n	images, then cropping them to thumbnails.'
	#
	# # bbox_file_name = 'samples/negative_unlabelled_info.dat'
	# neg_image_dir = 'samples/negative_unlabelled'
	# all_neg_images = glob.glob("{}/n*_*.*".format(neg_image_dir))
	# filter_pgm_prog = re.compile('{}/n\d*_\d*\.(pgm|svg)'.format(neg_image_dir))
	# filter_ext_prog = re.compile('{}/n\d*_\d*\.jpg'.format(neg_image_dir))
	# neg_images = filter(lambda x: filter_ext_prog.match(x) and not filter_pgm_prog.match(x), all_neg_images)
	# # for img in neg_images:
	# #	 findEllipsesInImage(img, bbox_file_name)
	#
	# # Load cache:
	# bbinfo_cache = loadBbinfo('negative_unlabelled_info')
	#
	# # Split neg_images into 8 parts:
	# numThreads = 8
	# neg_img_lists = [[] for i in range(numThreads)]
	# for i in range(len(neg_images)):
	# 	k = i % numThreads
	# 	neg_img_lists[k] += [neg_images[i]]
	#
	# def findEllipsesThread(img_list):
	# 	bbfn = 'bbinfo/negative_unlabelled_info__{}.dat'.format(random.randint(1000, 9999))
	# 	for img in img_list:
	# 		if img in bbinfo_cache:
	# 			print 'Skipping cached image: {}.'.format(img)
	# 			continue
	# 		findEllipsesInImage(img, bbfn)
	#
	# # with futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
	# # 	# Build set of futures:
	# # 	future_results = {}
	# # 	for img_list in neg_img_lists:
	# # 		future = executor.submit(findEllipsesThread, img_list)
	# # 		# future_results[future] = img_list
	# #
	# # 	for future in futures.as_completed(future_results):
	# # 		# search_word = future_results[future]
	# # 		if future.exception() is not None:
	# # 			print 'AN EXCEPTION OCCURRED: {}'.format(future.exception())
	# # 		else:
	# # 			print 'ELLIPSES FOR LIST CHUNK COMPLETE.'
	#
	# #
	# # # Use the bounding box data file to save cropped thumbnails of all ellipses:
	# # from extract_object_windows import extractObjectWindows
	# # for info_file in glob.glob('bbinfo/negative_unlabelled_info__*.dat'):
	# # 	print 'Processing info file:', info_file
	# # 	extractObjectWindows(info_file, (24, 24), 'samples/hard_negative')
	# # for info_file in glob.glob('bbinfo/info_dwsi__*.dat'):
	# # 	print 'Processing info file:', info_file
	# # 	extractObjectWindows(info_file, (24, 24), 'samples/hard_negative')
	#

	print '===== GENERATE TRIALS ====='
	from generate_trials import generateTrials
	generateTrials(args.template_yaml, args.output_dir)

	trial_files = glob.glob("{}/*.yaml".format(args.output_dir))

	print '===== VIEW SAMPLES ====='
	print 'Note: Comment out this section if you actually want to train anything.'
	for trial_yaml in trial_files:
		classifier_yaml = training.loadYamlFile(trial_yaml)
		output_dir = trial_yaml.split('.yaml')[0]
		training.viewPositiveSamples(classifier_yaml, output_dir)

	print '===== PREPROCESS TRIALS ====='

	preprocessing_was_successful = True
	maxImageCountDiff = (0, 0, 0)

	for trial_yaml in trial_files:
		print '    Preprocessing: {}'.format(trial_yaml)
		# Read classifier training file:
		classifier_yaml = training.loadYamlFile(trial_yaml)
		output_dir = trial_yaml.split('.yaml')[0]

		# Preprocess the trial:
		try:
			training.preprocessTrial(classifier_yaml, output_dir)
		except training.TooFewImagesError as e:
			preprocessing_was_successful = False
			print e
			imgCountDiff = map(lambda (p, r): r - p, zip(e.presentCounts, e.requiredCounts))
			maxImageCountDiff = map(lambda (m, c): max(m, c), zip(maxImageCountDiff, imgCountDiff))

	if not preprocessing_was_successful:
		print '\nNOT ENOUGH IMAGES! TRAINING CANCELLED!'
		print 'maxImageCountDiff: {}'.format(maxImageCountDiff)
		sys.exit(1)


	print '===== CREATE SAMPLES ====='

	for trial_yaml in trial_files:
		# Read classifier training file:
		classifier_yaml = training.loadYamlFile(trial_yaml)
		output_dir = trial_yaml.split('.yaml')[0]
		trained_classifier_xml = '{}/data/cascade.xml'.format(output_dir)

		if os.path.isfile(trained_classifier_xml):
			print '    Classifier already trained: {}'.format(trained_classifier_xml)
		else:
			print '    Creating samples for: {}'.format(trial_yaml)
			training.createSamples(classifier_yaml, output_dir)


	print '===== TRAIN CLASSIFIERS ====='

	def doTraining(fname):
		# Read classifier training file:
		classifier_yaml = training.loadYamlFile(fname)
		output_dir = fname.split('.yaml')[0]
		training.trainClassifier(classifier_yaml, output_dir)

	with futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
		future_results = dict((executor.submit(doTraining, fname), fname) for fname in trial_files)

		for future in futures.as_completed(future_results):
			fname = future_results[future]
			if future.exception() is not None:
				print '{} generated an exception: {}'.format(fname, future.exception())
			else:
				print '{} completed training successfully'.format(fname)


	# # TODO: Scan output files for possible errors.
	# # (check for bad words: 'cannot', 'error', 'not', 'fail')

	print '===== RUN CLASSIFIERS ====='

	# # TODO: Parallelise this code.
	# for trial_yaml in trial_files:
	# 	# Read classifier training file:
	# 	classifier_yaml = training.loadYamlFile(trial_yaml)
	# 	output_dir = trial_yaml.split('.yaml')[0]
	#
	# 	training.runClassifier(classifier_yaml, output_dir)

	def doRunning(trial_yaml):
		# Read classifier training file:
		classifier_yaml = training.loadYamlFile(trial_yaml)
		output_dir = trial_yaml.split('.yaml')[0]
		training.runClassifier(classifier_yaml, output_dir)

	synchronous = True
	if synchronous:
		for fname in trial_files:
			doRunning(fname)
	else:
		with futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
			future_results = dict((executor.submit(doRunning, fname), fname) for fname in trial_files)

			for future in futures.as_completed(future_results):
				fname = future_results[future]
				if future.exception() is not None:
					print '{} generated an exception: {}'.format(fname, future.exception())
				else:
					print '{} completed running successfully'.format(fname)


	print '===== COLLECT RESULTS ====='
