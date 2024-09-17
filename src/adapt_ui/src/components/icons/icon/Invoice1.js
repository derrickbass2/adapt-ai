import * as React from "react";

const SvgInvoice1 = (props) => (
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
            d="M5 5h14a3 3 0 0 1 3 3v8a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3V8a3 3 0 0 1 3-3m2 10h4a1 1 0 1 0 0-2H7a1 1 0 1 0 0 2m10 0h-2a1 1 0 1 1 0-2h2a1 1 0 1 1 0 2M4 9h16V8a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1z"
            clipRule="evenodd"
        />
    </svg>
);
export default SvgInvoice1;
