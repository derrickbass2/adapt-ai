import * as React from "react";

const SvgProduct = (props) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width={24}
        height={24}
        fill="none"
        {...props}
    >
        <path fill="#000" d="M0 0h24v24H0z" opacity={0.01}/>
        <path
            fill="#000"
            d="M11.25 11.83 3 8.36v7.73a1.69 1.69 0 0 0 1 1.52L11.19 21h.06zM12 10.5l8.51-3.57a1.6 1.6 0 0 0-.51-.38l-7.2-3.37a1.87 1.87 0 0 0-1.6 0L4 6.55a1.6 1.6 0 0 0-.51.38zM12.75 11.83V21h.05l7.2-3.39a1.69 1.69 0 0 0 1-1.51V8.36z"
        />
    </svg>
);
export default SvgProduct;
