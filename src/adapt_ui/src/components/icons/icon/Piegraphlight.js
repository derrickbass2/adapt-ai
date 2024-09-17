import * as React from "react";

const SvgPiegraphlight = (props) => (
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
            fillRule="evenodd"
            d="M21.17 10.33H14.5a.83.83 0 0 1-.83-.83V2.83A.83.83 0 0 1 14.5 2 7.5 7.5 0 0 1 22 9.5a.83.83 0 0 1-.83.83m-.9-1.66a5.83 5.83 0 0 0-4.94-4.94v4.94z"
            clipRule="evenodd"
        />
        <path
            fill="#fff"
            d="M21.08 12h-8.15a.91.91 0 0 1-.91-.91V2.92A.92.92 0 0 0 11 2a10 10 0 1 0 11 11 .92.92 0 0 0-.92-1"
        />
    </svg>
);
export default SvgPiegraphlight;
