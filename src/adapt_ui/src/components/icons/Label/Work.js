import * as React from "react";

const SvgWork = (props) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width={60}
        height={22}
        fill="none"
        {...props}
    >
        <rect width={60} height={22} fill="#FD9A56" opacity={0.2} rx={3}/>
        <path
            fill="#FD9A56"
            d="m18.782 15-2.964-8.46h1.272l2.268 6.66 2.328-6.66h.948l2.304 6.732 2.316-6.732h1.212L25.478 15h-1.044l-2.292-6.576L19.826 15zm12.67.108q-.887 0-1.535-.372a2.6 2.6 0 0 1-1.008-1.044q-.348-.684-.348-1.62t.348-1.608q.36-.684 1.008-1.056t1.536-.372q.864 0 1.512.372t1.008 1.056q.36.672.36 1.608t-.36 1.62q-.36.672-1.008 1.044t-1.512.372m0-.936q.745 0 1.2-.54.457-.54.457-1.56t-.456-1.548q-.456-.54-1.2-.54-.756 0-1.212.54-.456.528-.456 1.548t.456 1.56 1.212.54m4.242.828V9.144h1.176v1.044q.456-1.044 1.896-1.164L39.162 9l.084 1.032-.732.072q-1.596.156-1.596 1.644V15zm4.476 0V6.54h1.212v5.16h.024l2.544-2.556h1.5l-2.808 2.82L45.666 15h-1.512l-2.748-2.676h-.024V15z"
        />
    </svg>
);
export default SvgWork;