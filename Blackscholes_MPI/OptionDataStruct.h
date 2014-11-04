/*
 *   Author : Jahanzeb Maqbool | National University of Sciences and Technology | Islamabad | Pakistan
 */



#ifndef _OPTIONDATASTRUCT_H_
#define _OPTIONDATASTRUCT_H_

typedef struct OptionData_ {
        float s;          // spot price
        float strike;     // strike price
        float r;          // risk-free interest rate
        float divq;       // dividend rate
        float v;          // volatility
        float t;          // time to maturity or option expiration in years 
                          //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)  
        float divs;       // dividend vals (not used in this test)
        float DGrefval;   // DerivaGem Reference Value
	
	char OptionType;  // Option type.  "P"=PUT, "C"=CALL


} OptionData;

#endif




/*

int    * otype;
fptype * sptprice;
fptype * strike;
fptype * rate;
fptype * volatility;
fptype * otime;

buffer = (fptype *) malloc(5 * numOptions * sizeof(fptype) + PAD);
    sptprice = (fptype *) (((unsigned long long)buffer + PAD) & ~(LINESIZE - 1));
    strike = sptprice + numOptions;
    rate = strike + numOptions;
    volatility = rate + numOptions;
    otime = volatility + numOptions;

    buffer2 = (int *) malloc(numOptions * sizeof(fptype) + PAD);
    otype = (int *) (((unsigned long long)buffer2 + PAD) & ~(LINESIZE - 1));

    for (i=0; i<numOptions; i++) {
        otype[i]      = (data[i].OptionType == 'P') ? 1 : 0;
        sptprice[i]   = data[i].s;
        strike[i]     = data[i].strike;
        rate[i]       = data[i].r;
        volatility[i] = data[i].v;    
        otime[i]      = data[i].t;
    }




*/
