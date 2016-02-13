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

// // list of countries, defined with JavaScript object literals
// var countries = [
//   {"name": "Sweden"}, {"name": "China"}, {"name": "Peru"}, {"name": "Czech Republic"},
//   {"name": "Bolivia"}, {"name": "Latvia"}, {"name": "Samoa"}, {"name": "Armenia"},
//   {"name": "Greenland"}, {"name": "Cuba"}, {"name": "Western Sahara"}, {"name": "Ethiopia"},
//   {"name": "Malaysia"}, {"name": "Argentina"}, {"name": "Uganda"}, {"name": "Chile"},
//   {"name": "Aruba"}, {"name": "Japan"}, {"name": "Trinidad and Tobago"}, {"name": "Italy"},
//   {"name": "Cambodia"}, {"name": "Iceland"}, {"name": "Dominican Republic"}, {"name": "Turkey"},
//   {"name": "Spain"}, {"name": "Poland"}, {"name": "Haiti"}
// ];

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
        };
    },

    componentDidMount: function() {
        fetch('/_add_numbers', {
            method: "POST",
            body: {a: 3, b:4}
        })
        .then(function(response){
            console.log(response)
        });
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

    render: function() {
        return (
          <div>
            {// <input type>
            // <a href={this.state.lastGistUrl}>here</a>.
        }
          </div>
        );
      }

});


React.render(
  // <DynamicSearch items={ countries } />,
  <DetectorControl/>,
  document.getElementById('main')
);
