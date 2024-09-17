import * as React from "react";

const SvgCalendar1 = (props) => (
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
            d="M17 4h1a3 3 0 0 1 3 3v12a3 3 0 0 1-3 3H6a3 3 0 0 1-3-3V7a3 3 0 0 1 3-3h1V3a1 1 0 0 1 2 0v1h6V3a1 1 0 1 1 2 0zM7 16a1 1 0 1 0 2 0 1 1 0 0 0-2 0m9 1h-4a1 1 0 1 1 0-2h4a1 1 0 1 1 0 2M5 11h14V7a1 1 0 0 0-1-1h-1v1a1 1 0 1 1-2 0V6H9v1a1 1 0 0 1-2 0V6H6a1 1 0 0 0-1 1z"
            clipRule="evenodd"
        />
    </svg>
);
export default SvgCalendar1;
