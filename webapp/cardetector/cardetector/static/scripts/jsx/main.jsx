/** @jsx React.DOM */

// var DynamicSearch = React.createClass({
//
//   // sets initial state
//   getInitialState: function(){
//     return { searchString: '' };
//   },
//
//   // sets state, triggers render method
//   handleChange: function(event){
//     // grab value form input box
//     this.setState({searchString:event.target.value});
//     console.log("scope updated!")
//   },
//
//   render: function() {
//
//     var countries = this.props.items;
//     var searchString = this.state.searchString.trim().toLowerCase();
//
//     // filter countries list by value from input box
//     if(searchString.length > 0){
//       countries = countries.filter(function(country){
//         return country.name.toLowerCase().match( searchString );
//       });
//     }
//
//     return (
//       <div>
//         <input type="text" value={this.state.searchString} onChange={this.handleChange} placeholder="Search!" />
//         <ul>
//           { countries.map(function(country){ return <li>{country.name} </li> }) }
//         </ul>
//       </div>
//     )
//   }
//
// });


var React = require('react');
var ReactDOM = require('react-dom');
var Loader = require('react-loader');
var Select = require('react-select');
var _ = require('lodash');

// See: https://developer.mozilla.org/en-US/docs/Using_files_from_web_applications#Example_Using_object_URLs_to_display_images
function fetchFile(url, data) {
    if (data === undefined) {
        return fetch(url)
        .then(function(response) {
            return response.blob();
        }).then(function(response) {
            return URL.createObjectURL(response);
        });
    } else {
        return fetch(url, {
            method: 'POST',
            headers: { 'Content-type': 'application/json' },
            body: JSON.stringify(data)
        })
        .then(function(response) {
            return response.blob();
        }).then(function(response) {
            return URL.createObjectURL(response);
        });
    }
}

function fetchJson(url, data) {
    if (data === undefined) {
        return fetch(url)
        .then(function(response) {
            // console.log(response)
            return response.json();
        });
    } else {
        return fetch(url, {
            method: 'POST',
            headers: { 'Content-type': 'application/json' },
            body: JSON.stringify(data)
        })
        .then(function(response) {
            // console.log(response)
            return response.json();
        });
    }
}

var DetectorSettingsPanel = React.createClass({
    getInitialState: function() {
        return {
            imageDirValue: undefined,
            detectorDirValue: undefined,
        };
    },
    onDetectorChanged: function(input) {
        this.setState({'detectorDirValue': input})
        this.props.onDetectorChanged(input)
    },
    onImageDirectoryChanged: function(input) {
        this.setState({'imageDirValue': input})
        this.props.onImageDirectoryChanged(input)
    },
    getDetectorOptions: function(input) {
        return fetchJson('/_detector_directories')
        .then(function(json) {
            return {options: json['detector_directories']}
        })
    },
    getImageDirectoryOptions: function(input) {
        return fetchJson('/_image_directories')
        .then(function(json) {
            return {options: json['image_directories']}
        })
    },
    render: function() {
        // Note: Newer versions of react-select use the following syntax:
        // <Select.Async
        //     name='imageDir-select'
        //     loadOptions={this.getImageDirectoryOptions}
        //     onChange={this.changeImageDirectory}
        // />
        return (
        <div className="panel panel-default">
            <div className="panel-heading">
                <h3 className="panel-title">Settings</h3>
            </div>
            <div className="panel-body">
            {/* Forms reference: http://bootstrapdocs.com/v3.3.6/docs/css/#forms */}
            <form className="form-horizontal">
            <div className="form-group">
              <label className="col-sm-2 control-label">Detector</label>
              <div className="col-sm-10">
              <Select
                  name='detector-select'
                  value={this.state.detectorDirValue}
                  asyncOptions={this.getDetectorOptions}
                  onChange={this.onDetectorChanged}
              />
              </div>
            </div>
            <div className="form-group">
              <label className="col-sm-2 control-label">Test set</label>
              <div className="col-sm-10">
                <Select
                    name='imageDir-select'
                    value={this.state.imageDirValue}
                    asyncOptions={this.getImageDirectoryOptions}
                    onChange={this.onImageDirectoryChanged}
                />
              </div>
            </div>
            </form>
            </div>
        </div>
        );
    }
});

// Maintains a 16x9 aspect ratio.
// See: http://stackoverflow.com/a/12121309/3622526
var AspectContainer = React.createClass({
    render: function() {
        return (
            <div className="aspect16-9wrapper">
                <div className="aspect-inner">
                    {this.props.children}
                </div>
            </div>
        );
    }
});

var DetectorControl = React.createClass({
    getInitialPreviewState: function() {
        return {
            numImages: null,
            currentImgIndex: null,
            currentImgPath: null,
            currentImgUrl: null,
            detections: [],
        }
    },
    getInitialState: function() {
        var previewState = this.getInitialPreviewState();
        var state = {
            // Detector config:
            imageDir: null,
            detectorDir: null,
            autoDetectEnabled: false,

            // Extra:
            numParkingSpaces: null,
            loaded: true,
        };

        // Merge the state: http://stackoverflow.com/a/171256/3622526
        for (var attrname in previewState) {
            state[attrname] = previewState[attrname];
        }
        return state
    },
    componentDidMount: function() {
        // this.setState({liked: !this.state.liked});
        // $.getJSON('/_add_numbers', {
        //     a: 4,
        //     b: 6
        // }, function(data) {
        //     console.log(data)
        // });
        // fetchJson('/_add_numbers', {a:3, b:4, test:'test'})
        // .then(function(json) {
        //     console.log(json)
        // }).catch(function (error) {
        //     console.log('Request failed', error);
        // });
        // $.get(this.props.source,
        //     function (result) {
        //         var lastGist = result[0];
        //         this.setState({
        //             username: lastGist.owner.login,
        //             lastGistUrl: lastGist.html_url
        //         });
        //     }.bind(this)
        // );
    },
    updatePreviewState: function(options) {
        performDetection = undefined
        if (options !== undefined && options['performDetection']) {
            performDetection = options['performDetection']
        }
        if (performDetection === undefined) {
            performDetection = this.state.autoDetectEnabled
        }

        var invalid = !this.state.imageDir;
        invalid = invalid | (performDetection && !this.state.detectorDir);
        if (invalid) {
            console.log('Error: Detector and/or Test set are invalid.');
            return Promise.resolve();
        }

        this.setState({'loaded': false})
        args = {
            'currentImgIndex': this.state.currentImgIndex,
            'imageDir': this.state.imageDir,
            'detectorDir': this.state.detectorDir,
            'performDetection': performDetection,
        }

        // Get the new preview state:
        args['returnImage'] = false
        return fetchJson('/_update_preview_state', args)
        .then(function(json) {
            var previewKeys = Object.keys(this.getInitialPreviewState())
            var previewState = _.pick(json, previewKeys)
            this.setState(previewState)
            // this.setState({'loaded': true})
        }.bind(this))
        .catch(function(error) {
            this.setState({'loaded': true})
            console.log('updatePreviewState fetchJson, Request failed:', error);
        }.bind(this))
        .then(function() {
            // Get the new image:
            args['returnImage'] = true
            return fetchFile('/_update_preview_state', args)
            .then(function(url) {
                this.setState({'currentImgUrl': url})
                this.setState({'loaded': true})
            }.bind(this))
            .catch(function(error) {
                this.setState({'loaded': true})
                console.log('updatePreviewState fetchFile, Request failed:', error);
            }.bind(this))
        }.bind(this))
    },
    changeDetector: function(value) {
        console.log('changeDetector', value)
        this.setState({'detectorDir': value}, this.updatePreviewState)
    },
    changeImageDirectory: function(value) {
        console.log('changeImageDirectory', value)
        this.setState({'imageDir': value})
        this.setState(this.getInitialPreviewState(), this.updatePreviewState)
    },
    modifyImageIndex: function(val, func) {
        num = this.state.numImages
        index = this.state.currentImgIndex + val
        index = (index + num) % num
        this.setState({'currentImgIndex': index}, func)
    },
    moveToNextImage: function() {
        console.log('moveToNextImage')
        this.modifyImageIndex(1, this.updatePreviewState)
    },
    moveToPreviousImage: function() {
        console.log('moveToPreviousImage')
        this.modifyImageIndex(-1, this.updatePreviewState)
    },
    detectButtonClicked: function() {
        return this.updatePreviewState({'performDetection': true})
    },
    autoDetectChanged: function() {
        return this.setState({'autoDetectEnabled': !this.state.autoDetectEnabled})
    },
    render: function() {
        return (
        <div>
            <DetectorSettingsPanel
                onDetectorChanged={this.changeDetector}
                onImageDirectoryChanged={this.changeImageDirectory}
            />
{/*
<nav className="navbar navbar-default">
  <div className="container-fluid">
    <div className="navbar-header">
      <button type="button" className="navbar-toggle collapsed" data-toggle="collapse" data-target="#bs-example-navbar-collapse-1" aria-expanded="false">
        <span className="sr-only">Toggle navigation</span>
        <span className="icon-bar"></span>
        <span className="icon-bar"></span>
        <span className="icon-bar"></span>
      </button>
      <button className="btn btn-primary navbar-btn" onClick={this.detectButtonClicked}>Detect Cars</button>
    </div>
    <div className="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
      <ul className="nav navbar-nav">
        <li className="active"><p className="navbar-text">Image: {this.state.currentImgPath}</p></li>
        <li className="active"><p className="navbar-text">{this.state.detections.length} cars detected</p></li>
      </ul>
    </div>
  </div>
</nav>
*/}
            <form className='row'>
                <div className='col-sm-3 col-xs-6'>
                    <button type="button" className="btn btn-primary btn-block" onClick={this.detectButtonClicked}>
                        <span className="glyphicon glyphicon-search"></span> Detect Cars
                    </button>
                </div>
                <div className='col-sm-2 col-xs-6 checkbox'>
                    <label>
                      <input type="checkbox" checked={this.state.autoDetectEnabled} onChange={this.autoDetectChanged}/> Auto-detect
                    </label>
                </div>
                <div className='col-lg-5 col-md-4 col-sm-9 col-xs-12 text-left'>
                    <p>{this.state.currentImgPath}</p>
                </div>
                <div className='col-lg-2 col-md-3 col-sm-3 col-xs-12 text-right'>
                    <p>{this.state.detections.length} cars detected</p>
                </div>
            </form>
            <nav>
              <ul className="pager">
                <li className="previous" onClick={this.moveToPreviousImage}><a href="javascript:;"><span aria-hidden="true">&larr;</span> Previous Image</a></li>
                <li className="next" onClick={this.moveToNextImage}><a href="javascript:;">Next Image <span aria-hidden="true">&rarr;</span></a></li>
              </ul>
            </nav>
            <div className="jumbotron">
                <AspectContainer>
                    <div className="img-container">
                        <Loader loaded={this.state.loaded}>
                            {/* See: http://bootstrapdocs.com/v3.3.6/docs/css/#images-responsive */}
                            <img className="img-responsive center-block"
                                src={this.state.currentImgUrl}
                                onload={window.URL.revokeObjectURL(this.src)}
                            />
                        </Loader>
                    </div>
                </AspectContainer>
            </div>
        </div>
        );
    }
});

ReactDOM.render(
    // <DynamicSearch items={ countries } />,
    <DetectorControl/>,
    document.getElementById('main')
);
