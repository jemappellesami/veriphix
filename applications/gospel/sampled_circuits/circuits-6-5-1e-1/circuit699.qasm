OPENQASM 2.0;
include "qelib1.inc";
qreg q700[6];
cx q700[5],q700[4];
cx q700[3],q700[4];
cx q700[3],q700[2];
cx q700[2],q700[1];
cx q700[0],q700[1];
