import * as React from "react";

const SvgChat = (props) => (
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
            fillRule="evenodd"
            d="M6.378 3.692A10 10 0 0 1 19.07 4.93l.05.02a10 10 0 0 1-11 16.28 1.26 1.26 0 0 0-.64-.09L3.2 22H3a1 1 0 0 1-.73-.29A1 1 0 0 1 2 20.8l.88-4.23a1.06 1.06 0 0 0-.09-.64A10 10 0 0 1 6.378 3.692M7 12a1 1 0 1 0 2 0 1 1 0 0 0-2 0m5 1a1 1 0 1 1 0-2 1 1 0 0 1 0 2m3-1a1 1 0 1 0 2 0 1 1 0 0 0-2 0"
            clipRule="evenodd"
        />
    </svg>
);
export default SvgChat;
