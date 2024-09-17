(() => {
    'use strict';

    const fs = require('fs');
    const path = require('path');
    const SVGO = require('svgo');

    // Initialize SVGO with appropriate plugins
    const svgo = new SVGO({
        plugins: [
            {removeViewBox: false},
            {removeDimensions: true},
            {removeMetadata: true},
            {removeTitle: true},
            {removeDesc: true},
            {cleanupAttrs: true},
            {removeAttrs: {attrs: '(fill|stroke|style)'}},
            {convertPathData: true},
            {removeStyleElement: true}
        ]
    });

    // Function to optimize SVG file
    const optimizeSVG = (filePath) => {
        const svgData = fs.readFileSync(filePath, 'utf8');
        svgo.optimize(svgData, {path: filePath})
            .then(result => {
                if (result.error) {
                    console.error(`Error optimizing ${filePath}:`, result.error);
                } else {
                    fs.writeFileSync(filePath, result.data);
                    console.log(`Optimized: ${filePath}`);
                }
            })
            .catch(error => {
                console.error(`Error optimizing ${filePath}:`, error);
            });
    };

    // Function to recursively optimize all SVG files in a directory
    const optimizeSVGsInDirectory = (directory) => {
        fs.readdirSync(directory).forEach((file) => {
            const filePath = path.join(directory, file);
            const stats = fs.statSync(filePath);

            if (stats.isDirectory()) {
                optimizeSVGsInDirectory(filePath);
            } else if (stats.isFile() && path.extname(file) === '.svg') {
                optimizeSVG(filePath);
            }
        });
    };

    // Directories to optimize
    const directories = [
        '/Users/derrickbass/Public/adaptai/src/adapt_ui/src/assets/svgs',
        '/Users/derrickbass/Public/adaptai/src/adapt_ui/src/assets/svgs/Button',
        '/Users/derrickbass/Public/adaptai/src/adapt_ui/src/assets/svgs/dark_mode',
        '/Users/derrickbass/Public/adaptai/src/adapt_ui/src/assets/svgs/icon',
        '/Users/derrickbass/Public/adaptai/src/adapt_ui/src/assets/svgs/Label',
        '/Users/derrickbass/Public/adaptai/src/adapt_ui/src/assets/svgs/lite_mode',
        '/Users/derrickbass/Public/adaptai/src/adapt_ui/src/assets/svgs/Navigation',
    ];

    // Iterate through each directory and optimize the SVG files
    directories.forEach((directory) => {
        console.log(`Optimizing SVGs in: ${directory}`);
        optimizeSVGsInDirectory(directory);
    });

    console.log('SVG optimization completed.');
})();
