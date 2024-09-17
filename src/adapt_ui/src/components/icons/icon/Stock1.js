import * as React from "react";

const SvgStock1 = (props) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width={24}
        height={24}
        fill="none"
        {...props}
    >
        <path fill="#000" d="M0 0h24v24H0z" opacity={0.01}/>
        <path
            fill="#fff"
            d="M21 8V6a3 3 0 0 0-3-3H6a3 3 0 0 0-3 3v2zM3 10v8a3 3 0 0 0 3 3h5V10zM13 10v11h5a3 3 0 0 0 3-3v-8z"
        />
    </svg>
);
export default SvgStock1;
