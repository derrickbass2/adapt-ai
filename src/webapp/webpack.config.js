const path = require('path');
const webpack = require('webpack');
const SVGO_CONFIG = require('./config/svgo.config');

module.exports = {
    entry: './src/index.js',
    output: {
        path: path.resolve(__dirname, 'public'),
        filename: 'bundle.js',
        publicPath: '/', // Ensures that the correct public path is used
    },
    module: {
        rules: [
            {
                test: /\.(js|jsx)$/,
                exclude: /node_modules/,
                use: {
                    loader: 'babel-loader',
                    options: {
                        presets: ['@babel/preset-env', '@babel/preset-react'],
                    },
                },
            },
            {
                test: /\.css$/i,
                use: ['style-loader', 'css-loader'],
            },
            {
                test: /\.svg$/,
                use: [
                    {
                        loader: 'babel-loader',
                    },
                    {
                        loader: 'svg-url-loader',
                        options: SVGO_CONFIG,
                    },
                ],
            },
        ],
    },
    plugins: [
        new webpack.DefinePlugin({
            'process.env': {
                NODE_ENV: JSON.stringify(process.env.NODE_ENV || 'development'),
            },
        }),
    ],
    resolve: {
        extensions: ['.js', '.jsx'], // Allow import without specifying file extensions
    },
    devServer: {
        static: {
            directory: path.join(__dirname, 'public'),
        },
        historyApiFallback: true, // Ensures React Router works correctly
        port: 3000,
        proxy: {
            '/api': 'http://localhost:3001', // Proxy API requests to Express backend
        },
        hot: true, // Enable Hot Module Replacement
    },
};
