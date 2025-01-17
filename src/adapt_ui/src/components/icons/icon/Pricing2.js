import * as React from "react";

const SvgPricing2 = (props) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width={24}
        height={24}
        fill="none"
        {...props}
    >
        <path fill="#000" d="M0 0h24v24H0z" opacity={0.01}/>
        <path
            fill="#4379EE"
            fillRule="evenodd"
            d="m15.3 2.33 4.44 5A1 1 0 0 1 20 8v11.5a2.53 2.53 0 0 1-2.56 2.5H6.56A2.53 2.53 0 0 1 4 19.5v-15A2.53 2.53 0 0 1 6.56 2h8a1 1 0 0 1 .74.33M12 12H9a1 1 0 1 0 0 2h3a1 1 0 1 0 0-2m-3 6a1 1 0 1 1 0-2h6a1 1 0 1 1 0 2zm5-10.85a.79.79 0 0 0 .71.85h3.03L14 4z"
            clipRule="evenodd"
        />
    </svg>
);
export default SvgPricing2;
