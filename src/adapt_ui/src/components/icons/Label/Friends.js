import * as React from "react";

const SvgFriends = (props) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width={60}
        height={22}
        fill="none"
        {...props}
    >
        <rect width={60} height={22} fill="#D456FD" opacity={0.2} rx={3}/>
        <path
            fill="#D456FD"
            d="M10.984 15V6.54h5.376v1.008h-4.128v2.7h3.876v1.008h-3.876V15zm6.395 0V9.144h1.176v1.044q.456-1.044 1.896-1.164L20.847 9l.084 1.032-.732.072q-1.596.156-1.596 1.644V15zm4.369-7.212v-1.26h1.416v1.26zM21.856 15V9.144h1.212V15zm5.666.108q-1.44 0-2.268-.804-.828-.816-.828-2.22 0-.9.36-1.584a2.7 2.7 0 0 1 1.02-1.08q.648-.384 1.488-.384 1.212 0 1.908.78.696.768.696 2.124v.408h-4.296q.132 1.824 1.932 1.824.504 0 .984-.156.492-.156.924-.516l.36.84q-.396.36-1.02.564a4 4 0 0 1-1.26.204m-.18-5.22q-.756 0-1.2.468t-.54 1.248h3.276q-.036-.816-.432-1.26-.396-.456-1.104-.456M31.219 15V9.144h1.176v.972q.3-.528.816-.804a2.5 2.5 0 0 1 1.176-.276q2.076 0 2.076 2.352V15h-1.212v-3.54q0-.756-.3-1.104-.288-.348-.912-.348-.732 0-1.176.468-.432.456-.432 1.212V15zm9.146.108q-.768 0-1.356-.36a2.5 2.5 0 0 1-.9-1.056q-.324-.684-.324-1.62t.324-1.608a2.5 2.5 0 0 1 .9-1.056q.576-.372 1.356-.372.66 0 1.176.288t.768.78V6.54h1.212V15h-1.188v-1.02a1.86 1.86 0 0 1-.78.828q-.516.3-1.188.3m.312-.936q.744 0 1.2-.54t.456-1.56-.456-1.548q-.456-.54-1.2-.54-.756 0-1.212.54-.456.528-.456 1.548t.456 1.56 1.212.54m6.598.936q-.72 0-1.344-.18a3.1 3.1 0 0 1-1.044-.528l.348-.816q.444.312.972.48.54.168 1.08.168.636 0 .96-.228a.71.71 0 0 0 .324-.612q0-.312-.216-.48-.216-.18-.648-.276l-1.14-.228q-1.512-.312-1.512-1.56 0-.828.66-1.32t1.728-.492q.612 0 1.164.18.564.18.936.54l-.348.816a2.6 2.6 0 0 0-.828-.48 2.7 2.7 0 0 0-.924-.168q-.624 0-.948.24a.72.72 0 0 0-.324.624q0 .6.792.768l1.14.228q.78.156 1.176.528.408.372.408 1.008 0 .84-.66 1.32-.66.468-1.752.468"
        />
    </svg>
);
export default SvgFriends;