import React from 'react';
import DataProcessing from './components/DataProcessing';
import {BrowserRouter as Router, Route, Switch} from "react-router-dom";

const App = () => {
    return (
        <Router>
            <div>
                <h1>Welcome to the Data Processing App</h1>
                <Switch>
                    <Route exact path="/" component={DataProcessing}/>
                    {/* Other routes can go here */}
                </Switch>
            </div>
        </Router>
    );
};

export default App;