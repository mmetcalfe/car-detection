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

var DetectorControl = React.createClass({
    getInitialState: function() {
        return {
            // Detector config:
            imageDir: null,
            detectorDir: null,
            autoDetectEnabled: null,

            // Extra:
            numParkingSpaces: null,

            // State cache (the true state is on the server):
            numImages: null,
            currentImgIndex: null,
            currentImgPath: null,
            currentImg: null,
            detections: null,
            loaded: true,
        };
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

    changeDetector: function(value) {
        console.log('changeDetector', value)
    },
    changeImageDirectory: function(value) {
        console.log('changeImageDirectory', value)
    },
    moveToNextImage: function() {
        console.log('moveToNextImage')
    },
    moveToPreviousImage: function() {
        console.log('moveToPreviousImage')
    },
    detectButtonClicked: function() {
        console.log('detectButtonClicked')
    },
    render: function() {
        // Note: Newer versions of react-select use the following syntax:
        // <Select.Async
        //     name='imageDir-select'
        //     loadOptions={this.getImageDirectoryOptions}
        //     onChange={this.changeImageDirectory}
        // />
        return (
        <div>
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
                      asyncOptions={this.getDetectorOptions}
                      onChange={this.changeDetector}
                  />
                  </div>
                </div>
                <div className="form-group">
                  <label className="col-sm-2 control-label">Test set</label>
                  <div className="col-sm-10">
                    <Select
                        name='imageDir-select'
                        asyncOptions={this.getImageDirectoryOptions}
                        onChange={this.changeImageDirectory}
                    />
                  </div>
                </div>
                </form>
                </div>
            </div>
            <div className="jumbotron">
            </div>
            <nav>
              <ul className="pager">
                <li className="previous" onClick={this.moveToPreviousImage}><a href="#"><span aria-hidden="true">&larr;</span> Previous Image</a></li>
                <li className="next" onClick={this.moveToNextImage}><a href="#">Next Image <span aria-hidden="true">&rarr;</span></a></li>
              </ul>
            </nav>
            <button className="btn btn-primary" onClick={this.detectButtonClicked}>Detect Cars</button>
            <Loader loaded={this.state.loaded}>
            </Loader>
        </div>
        );
    }
});

ReactDOM.render(
    // <DynamicSearch items={ countries } />,
    <DetectorControl/>,
    document.getElementById('main')
);
