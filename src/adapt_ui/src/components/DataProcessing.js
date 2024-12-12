import React, {useState} from 'react';
import axios from 'axios';

const DataProcessing = () => {
    const [filePath, setFilePath] = useState('');
    const [featureCols, setFeatureCols] = useState('');
    const [numClusters, setNumClusters] = useState('');
    const [outputData, setOutputData] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);

        const requestData = {
            file_path: filePath,
            feature_cols: featureCols.split(','),
            num_clusters: numClusters
        };

        try {
            const response = await axios.post('http://localhost:3000/api/data', requestData, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('jwt_token')}` // Ensure JWT token is added
                }
            });
            setOutputData(response.data);
        } catch (error) {
            console.error('Error fetching data:', error);
            alert('Error occurred while processing data');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <h2>Data Processing</h2>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    placeholder="File Path"
                    value={filePath}
                    onChange={(e) => setFilePath(e.target.value)}
                    required
                />
                <input
                    type="text"
                    placeholder="Feature Columns (comma separated)"
                    value={featureCols}
                    onChange={(e) => setFeatureCols(e.target.value)}
                    required
                />
                <input
                    type="number"
                    placeholder="Number of Clusters"
                    value={numClusters}
                    onChange={(e) => setNumClusters(e.target.value)}
                    required
                />
                <button type="submit" disabled={loading}>Process Data</button>
            </form>
            {loading && <p>Loading...</p>}
            {outputData && (
                <div>
                    <h3>Processed Data:</h3>
                    <pre>{JSON.stringify(outputData, null, 2)}</pre>
                </div>
            )}
        </div>
    );
};

export default DataProcessing;