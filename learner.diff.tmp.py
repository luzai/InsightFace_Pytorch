
    def train_cotching_accbs(self, conf, epochs):
        self.model.train()
        self.model2.train()
        loader = self.loader
        self.evaluate_every = conf.other_every or len(loader) // 3
        self.save_every = conf.other_every or len(loader) // 3
        self.step = conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        imgs_l = []
        labels_l = []
        imgs2_l = []
        labels2_l = []
        accuracy = 0
        if conf.start_eval:
            for ds in ['cfp_fp', ]:
                accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                    conf,
                    self.loader.dataset.root_path,
                    ds)
                self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                logging.info(f'validation accuracy on {ds} is {accuracy} ')
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))
            while True:
                ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                if not imgs_l or sum([imgs.shape[0] for imgs in imgs_l]) < conf.batch_size:
                    imgs = data['imgs'].to(device=conf.model1_dev[0])
                    imgs2 = imgs
                    assert imgs.max() < 2
                    if 'labels_cpu' in data:
                        labels_cpu = data['labels_cpu'].cpu()
                    else:
                        labels_cpu = data['labels'].cpu()
                    labels = data['labels'].to(device=conf.model1_dev[0])
                    labels2 = labels
                    data_time.update(
                        lz.timer.since_last_check(verbose=False)
                    )
                    with torch.no_grad():
                        embeddings = self.model(imgs, mode='train')
                        embeddings2 = self.model(imgs2, mode='train')
                        thetas = self.head(embeddings, labels)
                        thetas2 = self.head2(embeddings2, labels2)
                        pred = thetas.argmax(dim=1)
                        pred2 = thetas2.argmax(dim=1)
                        disagree = pred != pred2
                        if disagree.sum().item() == 0:
                            continue  # todo this assert acc can finally reach bs
                        loss_xent = F.cross_entropy(thetas[disagree], labels[disagree], reduction='none')
                        loss_xent2 = F.cross_entropy(thetas2[disagree], labels2[disagree], reduction='none')
                        ind_sorted = loss_xent.argsort()
                        ind2_sorted = loss_xent2.argsort()
                        num_disagree = labels[disagree].shape[0]
                        # tau = 0.35
                        tau = 0.05
                        Ek = len(loader)
                        Emax = len(loader) * conf.epochs
                        lambda_e = 1 - min(self.step / Ek * tau, (1 + (self.step - Ek) / (Emax - Ek)) * tau)
                        num_remember = max(int(round(num_disagree * lambda_e)), 1)
                        ind_update = ind_sorted[:num_remember]
                        ind2_update = ind2_sorted[:num_remember]
                        imgs_l.append(imgs[ind_update].cpu())
                        labels_l.append(labels[ind_update].cpu())
                        imgs2_l.append(imgs[ind2_update].cpu())
                        labels2_l.append(imgs[ind2_update].cpu())
                        continue
                else:
                    imgs = torch.cat(imgs_l, dim=0)[:conf.batch_size].to(device=conf.model1_dev[0])
                    labels = torch.cat(labels_l, dim=0)[:conf.batch_size].to(device=conf.model1_dev[0])
                    labels_cpu = labels.cpu()
                    embeddings = self.model(imgs, mode='train')
                    thetas = self.head(embeddings, labels)
                    loss_xent = F.cross_entropy(thetas, labels)
                    self.optimizer.zero_grad()
                    if conf.fp16:
                        with amp.scale_loss(loss_xent, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss_xent.backward()
                    self.optimizer.step()

                    imgs2 = torch.cat(imgs2_l, dim=0)[:conf.batch_size].to(device=conf.model1_dev[0])
                    labels2 = torch.cat(labels2_l, dim=0)[:conf.batch_size].to(device=conf.model1_dev[0])
                    embeddings2 = self.model(imgs2, mode='train')
                    thetas2 = self.head(embeddings2, labels2)
                    loss_xent2 = F.cross_entropy(thetas2, labels2)
                    self.optimizer2.zero_grad()
                    if conf.fp16:
                        with amp.scale_loss(loss_xent2, self.optimizer2) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss_xent2.backward()
                    self.optimizer2.step()

                with torch.no_grad():
                    if conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, conf.dop)
                    if conf.mining == 'rand.id':
                        conf.dop[labels_cpu.numpy()] = 1
                conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/disagree', disagree.sum().item(), self.step)
                    writer.add_scalar('info/disagree_ratio', num_disagree / conf.batch_size, self.step)
                    writer.add_scalar('info/remenber', num_remember, self.step)
                    writer.add_scalar('info/remenber_ratio', num_remember / conf.batch_size, self.step)
                    writer.add_scalar('info/lambda_e', lambda_e, self.step)
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed', conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = conf.dop
                    if dop is not None:
                        writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(conf.explored == 0) / dop.shape[0], self.step)

                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                self.step += 1
                if conf.prof and self.step % 29 == 28:
                    break
            if conf.prof and e > 5:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')
