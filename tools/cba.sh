#!/usr/bin/env bash
set -x # set -e

for i in ft4.mg4.cyc1.mom.lm10 ft4.mg4.cyc.swa.mom.lm10 ft3.mg4.cyc1  ft3.mg4.cyc7  ft3.mg4.cyc.swa  ft3.mg4.cyc.swa.mod  ft3.mg4.cyc1.gm1  ft3.mg4.cyc1.gm9  ft3.mg4.cyc1.gm1.swa  ft3.mg4.cyc1.mom ft3.mg4.cyc1.1en4  ft3.mg4.cyc1.5en4  
do
echo $i
cb.sh $i /home/zl/prj/fcpth.r100.bs
done

exit

for i in  ft4.mg4.dop.222  ft4.mg4.dop.444  ft4.mg4.dop.555  ft4.mg4.fromfea.zjjk.lm10  ft4.tri9.ct10.inst8  ft4.mg4.fromfea.zjjk  ft4.tri9.ct10.inst16  ft4.mg4.fromfea.zjjk.lm10.ld2.decay10  ft4.tri9.ct10.p.5.n2.inst8.decay10  ft4.tri9.ct10.p.5.n2.inst8  ft4.tri9.ct10.inst8.decay10  ft4.tri9.ct10.p.5.n4.inst8.decay10  ft4.tri9.ct10.p.5.n4.inst8  ft4.tri9.ct10.p.5.n2.inst8.dop  ft4.tri9.ct10.p.5.n4.inst8.dop  ft4.mg4.cyc.swa.mod.mom.lm10  ft4.tri9.ct10.p.5.n2.inst8.dop.decay10  ft4.mg4.cyc1.mom.lm10  ft4.tri9.ct10.p.5.n2.inst8.dop333  ft4.tri9.ct10.p.5.n4.inst8.dop.decay10  ft4.tri9.ct10.p.5.n4.inst8.dop333 ft4.mg4.cyc.swa.mom.lm10
do
echo $i 
./cb.sh $i /home/zl/prj/fcpth.r100.bs 
done

for i in ft3.tri9.ct10.inst8 ft3.tri9.ct10.tris1.p2.n50 ft3.tri9.ct10.tris1 ft3.tri9 ft3.tri9 ft3.neg.48.02 ft3.mg4.dop.333 ft3.mg4.ct10 ft3.tri9.ct10.tris1.p2.n50.dop ft3.neg.25.1.dop ft3.tri9.ct10.tris1.p2.n40 ft3.neg.25.1 ft3.mg4.decay10 ft3.mg4.dop.111 ft3.neg.48.02.dop ft3.tri9.ct10.tris64.p1.n2 ft3.mg4.fromwei ft3.mg5 ft3.mg4.wd5en4.wm1 ft3.tri9.ct10.tris1.p2.n50 ft3.mg4 ft3.mg4.wd5en4.wm1.lm10 ft3.tri9.ct10 ft3.tri9.ct10.inst8 ft3.tri9.ct10.tris1 ft3.mg4.decay2 ft3.mg4.dop.162 ft3.mg4.wd5en4 ft2.neg.25.1.dop.ct10 ft3.mg45 ft3.mg4.fromfea.10 ft3.tri9.ct10 ft3.tri9.ct10.tris1.p2.n40 ft3.tri9.ct10.tris1.p2.n50.dop ft3.tri9.ct10.tris64.p1.n2 mg4.nochsfst
do 
echo $i 
./cb.sh $i /home/zl/prj/fcpth.r100.bs 
done

for i in ft3.mg4.ct10.ipabn ft3.mg4.ct10.ipabn.headp
do
echo $i 
./cb.sh $i /home/zl/prj/arc/fcpth.for.r100.ipabn
done 

./cb.sh ft2.mg4.tri.1.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft2.neg.2.2.top10.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn

./cb.sh ft2.dop.111.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft2.dop.162.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft2.dop.333.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft2.mg4.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn

./cb.sh ft2.mg4.ct15.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft2.mg4.ct5.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft2.mg4.tri.5.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft2.mg4.tri.9.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn

./cb.sh ft2.neg.25.1.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft2.neg.25.1.dop.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft2.neg.25.1.dop.tri.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft2.neg.25.25.top10.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn

./cb.sh ft2.neg.3.1.top10.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft2.neg.3.2.top10.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft2.neg.48.02.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft2.neg.48.02.dop.ct10.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn

./cb.sh r100.alpha.elu.in.1e-1 /home/zl/prj/fcpth.r100.ranger
./cb.sh r100.alpha.neg.25.1.top3 /home/zl/prj/fcpth.r100.bs 
./cb.sh r100.neg.3.1.top3 /home/zl/prj/fcpth.r50.bs 
./cb.sh r100.neg.48.02.top3 /home/zl/prj/fcpth.r50.bs  
./cb.sh r100.dop.6.27 /home/zl/prj/fcpth.r50.bs  
./cb.sh r100.dop.9.27 /home/zl/prj/fcpth.r50.bs  
./cb.sh r100.neg.48.02.top3.dop /home/zl/prj/fcpth.r50.bs  
./cb.sh r100.ds.bl.5 /home/zl/prj/fcpth.ds.bl.5  
./cb.sh r100.ds.bl.9 /home/zl/prj/fcpth.ds.bl.5  
./cb.sh r100.ds.bl /home/zl/prj/fcpth.ds.bl  
./cb.sh r100.neg.2.2.top3 /home/zl/prj/fcpth.r50.bs  
./cb.sh r100.neg.25.1.top3.dop /home/zl/prj/fcpth.r50.bs  
./cb.sh r100.bs.pt /home/zl/prj/fcpth.r100.bs  
./cb.sh r100.alpha.neg.25.1.top3.pt /home/zl/prj/fcpth.r100.bs   
./cb.sh r100.alpha.neg.25.1.top3.pt.cont /home/zl/prj/fcpth.r100.bs  
./cb.sh r100.bs.2.cont /home/zl/prj/fcpth.r100.bs  
./cb.sh alphaf64.dop.tri.chsfst.ft.4 /home/zl/prj/fcpth.ft   # todo fcpth --> fcpth.ft 
./cb.sh r100.dop.9.27.fp16 /home/zl/prj/fcpth.r100.bs  
./cb.sh ft.bs /home/zl/prj/fcpth.r100.bs  
./cb.sh ft.dop.162 /home/zl/prj/fcpth.r100.bs  
./cb.sh ft.neg.25.1.dop /home/zl/prj/fcpth.r100.bs
./cb.sh ft.in /home/zl/prj/fcpth.in 
./cb.sh r100.bs.fp16 /home/zl/prj/fcpth.r100.bs  
./cb.sh r100.bs.in /home/zl/prj/fcpth.in  
./cb.sh r100.neg.25.1.top3.dop.pt.fp16 /home/zl/prj/fcpth.r100.bs  
./cb.sh alphaf64.r100.arc.neg /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh alphaf64.r100.arc.neg.2.15 /home/zl/prj/fcpth.r100.bs
./cb.sh alphaf64.r100.arc.neg.25.25 /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft.ipabn.headp.decay10 /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft.neg.25.1 /home/zl/prj/fcpth.r100.bs
./cb.sh ft.in /home/zl/prj/fcpth.in
./cb.sh ft.neg.25.1.dop /home/zl/prj/fcpth.r100.bs
./cb.sh ft.neg.48.02.dop /home/zl/prj/fcpth.r100.bs
./cb.sh ft.bs.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft.ipabn /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft.ipabn.headp /home/zl/prj/arc/fcpth.for.r100.ipabn
./cb.sh ft.ipabn.headp.decay10 /home/zl/prj/arc/fcpth.for.r100.ipabn

./cb.sh r50.ranger.ct10 /home/zl/prj/fcpth.r50.ranger.84
