import * as React from "react";

const SvgDark = (props) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        xmlnsXlink="http://www.w3.org/1999/xlink"
        width={1201}
        height={70}
        fill="none"
        {...props}
    >
        <path fill="#273142" d="M0 0h1201v70H0z"/>
        <path
            stroke="#CCC"
            strokeWidth={0.2}
            d="M1161 44.1c5.03 0 9.1-4.074 9.1-9.1s-4.07-9.1-9.1-9.1-9.1 4.074-9.1 9.1 4.07 9.1 9.1 9.1Z"
        />
        <path
            fill="#fff"
            d="m1161 35.793-2.27-2.647c-.17-.195-.44-.195-.6 0a.554.554 0 0 0 0 .708l2.57 3a.38.38 0 0 0 .6 0l2.57-3a.554.554 0 0 0 0-.708c-.16-.195-.43-.195-.6 0z"
        />
        <mask
            id="Dark_svg__a"
            width={6}
            height={4}
            x={1158}
            y={33}
            maskUnits="userSpaceOnUse"
            style={{maskType: "luminance"}}
        >
            <path
                fill="#fff"
                d="m1161 35.793-2.27-2.647c-.17-.195-.44-.195-.6 0a.554.554 0 0 0 0 .708l2.57 3a.38.38 0 0 0 .6 0l2.57-3a.554.554 0 0 0 0-.708c-.16-.195-.43-.195-.6 0z"
            />
        </mask>
        <path
            fill="#fff"
            d="M1066.11 30v-9.87h1.49l3.5 6.454 3.49-6.454h1.47V30h-1.62v-6.65l-2.83 5.152h-1.05l-2.83-5.124V30zm15.08.14c-.71 0-1.33-.145-1.85-.434s-.93-.7-1.22-1.232c-.29-.541-.43-1.176-.43-1.904s.14-1.358.43-1.89.7-.943 1.22-1.232 1.14-.434 1.85-.434 1.32.145 1.84.434a3 3 0 0 1 1.22 1.232c.29.532.44 1.162.44 1.89s-.15 1.363-.44 1.904a3 3 0 0 1-1.22 1.232c-.52.29-1.13.434-1.84.434m0-1.33q.78 0 1.26-.56c.31-.383.47-.943.47-1.68 0-.747-.16-1.302-.47-1.666q-.48-.56-1.26-.56-.795 0-1.26.56-.48.547-.48 1.666 0 1.105.48 1.68.465.56 1.26.56m4.94 1.19v-6.846h1.71v1.064c.23-.392.54-.69.94-.896.4-.205.85-.308 1.34-.308 1.62 0 2.42.938 2.42 2.814V30h-1.75v-4.088c0-.532-.1-.92-.31-1.162-.19-.243-.5-.364-.92-.364q-.765 0-1.23.49-.45.476-.45 1.274V30zm8.08-8.316v-1.708h1.96v1.708zm.11 8.316v-6.846h1.75V30zm7.51 0v-9.87h4.28c1.09 0 1.92.261 2.51.784.59.513.88 1.237.88 2.17q0 1.106-.57 1.834c-.38.476-.94.798-1.65.966q.705.225 1.17 1.078L1110.1 30h-1.97l-1.71-3.15q-.255-.462-.6-.63-.33-.168-.87-.168h-1.33V30zm1.79-5.278h2.19c1.3 0 1.96-.532 1.96-1.596 0-1.055-.66-1.582-1.96-1.582h-2.19zm10.58 5.418c-.71 0-1.32-.145-1.84-.434a3 3 0 0 1-1.22-1.232c-.29-.541-.44-1.176-.44-1.904s.15-1.358.44-1.89a3 3 0 0 1 1.22-1.232c.52-.29 1.13-.434 1.84-.434s1.33.145 1.85.434.93.7 1.22 1.232.43 1.162.43 1.89-.14 1.363-.43 1.904c-.29.532-.7.943-1.22 1.232s-1.14.434-1.85.434m0-1.33q.795 0 1.26-.56.48-.575.48-1.68 0-1.12-.48-1.666-.465-.56-1.26-.56-.78 0-1.26.56c-.31.364-.47.92-.47 1.666 0 .737.16 1.297.47 1.68q.48.56 1.26.56m5.43 3.71 1.26-2.8-2.84-6.566h1.86l1.92 4.788 1.96-4.788h1.76l-4.12 9.366zM1065.13 50l3.77-8.46h1.03l3.78 8.46h-1.27l-.89-2.064h-4.28l-.88 2.064zm4.27-7.08-1.69 4.008h3.42l-1.7-4.008zm7.46 7.188q-.765 0-1.35-.36a2.4 2.4 0 0 1-.9-1.056q-.33-.684-.33-1.62t.33-1.608q.315-.684.9-1.056.57-.372 1.35-.372c.44 0 .83.096 1.18.288.34.192.6.452.77.78V41.54h1.21V50h-1.19v-1.02q-.255.528-.78.828c-.34.2-.74.3-1.19.3m.31-.936q.75 0 1.2-.54c.31-.36.46-.88.46-1.56s-.15-1.196-.46-1.548q-.45-.54-1.2-.54c-.5 0-.9.18-1.21.54q-.45.528-.45 1.548t.45 1.56c.31.36.71.54 1.21.54m4.57.828v-5.856h1.18v.948c.18-.336.42-.596.73-.78s.68-.276 1.09-.276q1.35 0 1.74 1.176c.19-.368.45-.656.8-.864q.51-.312 1.17-.312c1.31 0 1.97.784 1.97 2.352V50h-1.21v-3.552q0-.744-.27-1.092-.24-.348-.84-.348c-.44 0-.78.156-1.04.468-.26.304-.38.72-.38 1.248V50h-1.22v-3.552c0-.496-.08-.86-.25-1.092q-.255-.348-.84-.348c-.44 0-.79.156-1.04.468-.25.304-.37.72-.37 1.248V50zm10.27-7.212v-1.26h1.41v1.26zm.1 7.212v-5.856h1.22V50zm2.92 0v-5.856h1.18v.972q.3-.528.81-.804c.36-.184.75-.276 1.18-.276 1.38 0 2.08.784 2.08 2.352V50h-1.22v-3.54q0-.756-.3-1.104c-.19-.232-.49-.348-.91-.348q-.735 0-1.17.468c-.29.304-.44.708-.44 1.212V50z"
        />
        <path
            fill="#D8D8D8"
            fillRule="evenodd"
            d="M1023 57c12.15 0 22-9.85 22-22s-9.85-22-22-22-22 9.85-22 22 9.85 22 22 22"
            clipRule="evenodd"
        />
        <mask
            id="Dark_svg__b"
            width={44}
            height={44}
            x={1001}
            y={13}
            maskUnits="userSpaceOnUse"
            style={{maskType: "luminance"}}
        >
            <path
                fill="#fff"
                fillRule="evenodd"
                d="M1023 57c12.15 0 22-9.85 22-22s-9.85-22-22-22-22 9.85-22 22 9.85 22 22 22"
                clipRule="evenodd"
            />
        </mask>
        <g mask="url(#Dark_svg__b)">
            <path fill="url(#Dark_svg__c)" d="M999 8h50v54h-50z"/>
        </g>
        <path
            fill="#F2F2F2"
            d="M908.148 41v-9.87h6.384v1.176h-4.984V35.4h4.69v1.176h-4.69v3.248h4.984V41zm8.09 0v-6.832h1.372v1.134a2.36 2.36 0 0 1 .952-.938 2.9 2.9 0 0 1 1.372-.322q2.422 0 2.422 2.744V41h-1.414v-4.13q0-.882-.35-1.288-.336-.406-1.064-.406-.854 0-1.372.546-.504.532-.504 1.414V41zm11.161 2.646q-.924 0-1.722-.224a4 4 0 0 1-1.414-.672l.434-1.008q.615.42 1.26.616.644.195 1.344.196 1.96 0 1.96-1.96v-1.078q-.294.616-.924.966-.616.35-1.386.35-.924 0-1.61-.42a2.9 2.9 0 0 1-1.064-1.19q-.378-.77-.378-1.792 0-1.008.378-1.764.378-.77 1.064-1.19.686-.434 1.61-.434.783 0 1.4.35.615.35.91.966v-1.19h1.372v6.258q0 1.61-.826 2.408-.827.812-2.408.812m-.112-3.92q.895 0 1.428-.616.531-.615.532-1.68 0-1.065-.532-1.666-.533-.615-1.428-.616-.897 0-1.428.616-.532.602-.532 1.666t.532 1.68q.531.615 1.428.616m7.556 1.4q-1.12 0-1.68-.63-.546-.63-.546-1.848V31.13h1.414v7.434q0 1.386 1.134 1.386.168 0 .308-.014.154-.014.308-.056l-.028 1.134a4 4 0 0 1-.91.112m1.927-8.54v-1.47h1.652v1.47zm.126 8.414v-6.832h1.414V41zm5.799.126q-.84 0-1.568-.21-.729-.225-1.218-.616l.406-.952q.518.365 1.134.56.63.195 1.26.196.741 0 1.12-.266a.83.83 0 0 0 .378-.714q0-.365-.252-.56-.252-.21-.756-.322l-1.33-.266q-1.764-.365-1.764-1.82 0-.966.77-1.54t2.016-.574q.713 0 1.358.21.657.21 1.092.63l-.406.952a3 3 0 0 0-.966-.56 3.2 3.2 0 0 0-1.078-.196q-.728 0-1.106.28a.84.84 0 0 0-.378.728q0 .7.924.896l1.33.266q.91.182 1.372.616.476.434.476 1.176 0 .98-.77 1.54-.77.546-2.044.546m4.387-.126v-9.87h1.414v4.102q.35-.588.938-.882a2.9 2.9 0 0 1 1.344-.308q2.422 0 2.422 2.744V41h-1.414v-4.13q0-.882-.35-1.288-.336-.406-1.064-.406-.854 0-1.372.546-.504.532-.504 1.414V41z"
        />
        <path
            fill="#D5D5D5"
            d="m970 34.925-3.087-3.087a.583.583 0 0 0-.825.825l3.5 3.5a.583.583 0 0 0 .825 0l3.5-3.5a.583.583 0 0 0-.825-.825z"
        />
        <mask
            id="Dark_svg__d"
            width={10}
            height={6}
            x={965}
            y={31}
            maskUnits="userSpaceOnUse"
            style={{maskType: "luminance"}}
        >
            <path
                fill="#fff"
                d="m970 34.925-3.087-3.087a.583.583 0 0 0-.825.825l3.5 3.5a.583.583 0 0 0 .825 0l3.5-3.5a.583.583 0 0 0-.825-.825z"
            />
        </mask>
        <rect width={40} height={27} x={851} y={22} fill="#D8D8D8" rx={5}/>
        <mask
            id="Dark_svg__e"
            width={40}
            height={27}
            x={851}
            y={22}
            maskUnits="userSpaceOnUse"
            style={{maskType: "luminance"}}
        >
            <rect width={40} height={27} x={851} y={22} fill="#fff" rx={5}/>
        </mask>
        <g mask="url(#Dark_svg__e)">
            <path fill="url(#Dark_svg__f)" d="M851 22h40v27h-40z"/>
        </g>
        <path
            fill="#4880FF"
            fillRule="evenodd"
            d="M806.028 24a4.5 4.5 0 0 0-4.473 4.003L800.5 37.5h-3A1.5 1.5 0 0 0 796 39v1.5a1.5 1.5 0 0 0 1.5 1.5h21a1.5 1.5 0 0 0 1.5-1.5V39a1.5 1.5 0 0 0-1.5-1.5h-3l-1.055-9.497A4.5 4.5 0 0 0 809.972 24z"
            clipRule="evenodd"
        />
        <rect
            width={6}
            height={6}
            x={805}
            y={43.5}
            fill="#fff"
            opacity={0.9}
            rx={2.25}
        />
        <path
            fill="#FF4F4F"
            fillOpacity={0.6}
            fillRule="evenodd"
            d="M817 36a9 9 0 0 0 9-9 9 9 0 0 0-9-9 9 9 0 0 0-9 9 9 9 0 0 0 9 9"
            clipRule="evenodd"
            opacity={0.173}
        />
        <path
            fill="#FF4F4F"
            fillRule="evenodd"
            d="M817 35a8 8 0 1 0 0-16 8 8 0 0 0 0 16"
            clipRule="evenodd"
        />
        <path
            fill="#fff"
            d="M816.912 31.12q-1.572 0-2.436-1.104-.852-1.104-.852-3.096 0-2.148.96-3.324.972-1.176 2.676-1.176.672 0 1.332.252.66.24 1.128.696l-.504 1.152a3 3 0 0 0-.948-.6 2.7 2.7 0 0 0-1.032-.204q-1.044 0-1.596.732t-.552 2.184v.348q.252-.66.816-1.032a2.3 2.3 0 0 1 1.296-.372q.732 0 1.296.348t.888.96.324 1.404q0 .816-.36 1.464a2.6 2.6 0 0 1-.984 1.008q-.624.36-1.452.36m-.084-1.224q.648 0 1.044-.42.408-.432.408-1.128t-.408-1.128q-.396-.432-1.044-.432t-1.056.432q-.396.432-.396 1.128t.396 1.128q.408.42 1.056.42"
        />
        <rect width={388} height={38} x={78} y={16} fill="#323D4E" rx={19}/>
        <rect
            width={387.4}
            height={37.4}
            x={78.3}
            y={16.3}
            stroke="#CFCFCF"
            strokeOpacity={0.114}
            strokeWidth={0.6}
            rx={18.7}
        />
        <path
            fill="#fff"
            d="M127.479 40.126q-1.161 0-2.1-.294a4.9 4.9 0 0 1-1.624-.882l.406-.924q.715.546 1.498.812.785.266 1.82.266 1.26 0 1.862-.462a1.5 1.5 0 0 0 .616-1.246q0-.615-.448-.98-.434-.364-1.47-.574l-1.456-.294q-1.315-.28-1.974-.91-.644-.645-.644-1.708 0-.882.448-1.54.462-.659 1.274-1.022.813-.364 1.876-.364.995 0 1.848.322a3.85 3.85 0 0 1 1.442.896l-.406.896a4 4 0 0 0-1.33-.826 4.4 4.4 0 0 0-1.568-.266q-1.106 0-1.764.518a1.61 1.61 0 0 0-.658 1.344q0 .672.406 1.064.42.392 1.358.574l1.456.308q1.414.294 2.086.896.686.588.686 1.61 0 .825-.448 1.456-.433.63-1.26.98-.811.35-1.932.35m8.334 0q-1.61 0-2.548-.938-.939-.952-.938-2.576 0-1.05.42-1.848.42-.812 1.148-1.246.742-.448 1.708-.448 1.386 0 2.17.896.783.882.784 2.436v.434h-5.124q.055 1.162.672 1.778.615.602 1.708.602.615 0 1.176-.182.56-.195 1.064-.63l.392.798q-.462.434-1.176.686a4.6 4.6 0 0 1-1.456.238m-.182-6.216q-.966 0-1.526.602t-.658 1.582h4.13q-.042-1.035-.546-1.61-.49-.574-1.4-.574m6.709 6.216q-.7 0-1.26-.266a2.3 2.3 0 0 1-.868-.756 1.85 1.85 0 0 1-.322-1.064q0-.742.378-1.176.392-.435 1.274-.616.895-.195 2.436-.196h.448v-.434q0-.868-.364-1.246-.351-.392-1.134-.392-.617 0-1.19.182a4.5 4.5 0 0 0-1.176.56l-.392-.826a4.3 4.3 0 0 1 1.288-.602 5.3 5.3 0 0 1 1.47-.224q1.316 0 1.946.644.644.645.644 2.002V40h-1.064v-1.176a2.1 2.1 0 0 1-.826.952q-.547.35-1.288.35m.182-.854q.84 0 1.372-.574.531-.588.532-1.484v-.42h-.434q-1.135 0-1.792.112-.645.098-.91.364-.252.252-.252.7 0 .574.392.938.405.364 1.092.364m5.1.728v-6.804h1.106v1.218q.546-1.232 2.24-1.358l.406-.042.084.98-.714.084q-.966.084-1.47.616-.504.517-.504 1.428V40zm7.819.126q-1.022 0-1.764-.434a3.02 3.02 0 0 1-1.148-1.246q-.392-.811-.392-1.89 0-1.624.882-2.548.882-.938 2.422-.938.63 0 1.246.224.616.225 1.022.63l-.392.826a2.6 2.6 0 0 0-.896-.574 2.6 2.6 0 0 0-.91-.182q-1.064 0-1.638.672-.574.658-.574 1.904 0 1.218.574 1.932.575.7 1.638.7.434 0 .91-.182.49-.182.896-.588l.392.826q-.406.405-1.036.644-.616.224-1.232.224m3.597-.126v-9.87h1.134v4.186q.335-.616.938-.924.615-.322 1.372-.322 2.422 0 2.422 2.688V40h-1.134v-4.172q0-.953-.378-1.386-.364-.448-1.176-.448-.939 0-1.498.588-.546.574-.546 1.54V40z"
            opacity={0.8}
        />
        <g
            stroke="#fff"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.2}
            opacity={0.8}
        >
            <path
                d="M103.694 38.535a5.371 5.371 0 1 0-4.204-9.886 5.371 5.371 0 0 0 4.204 9.886"
                clipRule="evenodd"
            />
            <path d="m105.39 37.39 4.166 4.166"/>
        </g>
        <defs>
            <pattern
                id="Dark_svg__c"
                width={1}
                height={1}
                patternContentUnits="objectBoundingBox"
            >
                <use xlinkHref="#Dark_svg__g" transform="scale(.00067)"/>
            </pattern>
            <pattern
                id="Dark_svg__f"
                width={1}
                height={1}
                patternContentUnits="objectBoundingBox"
            >
                <use xlinkHref="#Dark_svg__h" transform="scale(.0125 .025)"/>
            </pattern>
            <image
                id="Dark_svg__g"
                width={1500}
                height={1500}
            />
            <image
                xlinkHref="data:image/png;base64,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"
                id="Dark_svg__h"
                width={80}
                height={40}
            />
        </defs>
    </svg>
);
export default SvgDark;